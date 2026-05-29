import itertools
import re
import unicodedata
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .cache import (
    check_cache,
    count_corrections,
    find_similar_correction,
    memory_stats,
    save_correction,
)
from .model import DEFAULT_BATCH_SIZE, detect_pairs
from .rl_agent import decide, policy_stats, update


MAX_SEGMENTS = 120
MAX_COMPARISONS = 180
MAX_RETURNED_COMPARISONS = 48
MAX_RETURNED_CONFLICTS = 24
MIN_SEGMENT_WORDS = 3
CLAUSE_SPLIT_WORDS = 16
LOCAL_WINDOW = 3
CONTRADICTION_ALERT_THRESHOLD = 0.6
NEUTRAL_MARGIN = 0.15
CLAUSE_CONNECTOR_PATTERN = re.compile(
    r"\s+(?:ya que|debido a que|porque|aunque|sin embargo|por otro lado|pero|mientras que|en cambio)\s+",
    flags=re.IGNORECASE,
)
DISCOURSE_ONLY_PHRASES = {
    "debido a que",
    "en cambio",
    "por otro lado",
    "sin embargo",
    "ya que",
}

STOPWORDS = {
    "ademas",
    "ante",
    "aqui",
    "cada",
    "como",
    "con",
    "contra",
    "cuando",
    "del",
    "desde",
    "donde",
    "durante",
    "el",
    "ellos",
    "ella",
    "ellas",
    "este",
    "esta",
    "estos",
    "estas",
    "fue",
    "han",
    "hay",
    "las",
    "la",
    "lo",
    "los",
    "mas",
    "para",
    "pero",
    "por",
    "porque",
    "que",
    "segun",
    "ser",
    "sin",
    "sobre",
    "son",
    "sus",
    "tambien",
    "tiene",
    "tienen",
    "una",
    "un",
    "uno",
    "unos",
    "unas",
}


class FeedbackRequest(BaseModel):
    confidence: float
    action: int
    accepted: bool
    correct_label: Optional[str] = None
    premise: Optional[str] = None
    hypothesis: Optional[str] = None
    context: Optional[dict] = None


class ContradictionRequest(BaseModel):
    premise: str
    hypothesis: str


class DocumentRequest(BaseModel):
    text: str
    max_pairs: Optional[int] = None


app = FastAPI(title="Detector de Contradicciones IA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _label_key(label: str) -> str:
    return str(label).lower()


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)
    return {
        token
        for token in re.findall(r"\b[a-z0-9]{3,}\b", normalized)
        if token not in STOPWORDS
    }


def _entities(text: str) -> set[str]:
    entities = set()
    for word in re.findall(r"\b\w+\b", text, flags=re.UNICODE):
        if len(word) > 2 and word[0].isupper():
            normalized = _normalize_text(word)
            if normalized not in STOPWORDS:
                entities.add(normalized)
    return entities


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _extract_paragraphs(text: str) -> list[dict]:
    paragraphs = []
    current_start = None
    current_end = None
    position = 0

    for line in text.splitlines(True):
        if line.strip():
            if current_start is None:
                current_start = position
            current_end = position + len(line)
        elif current_start is not None and current_end is not None:
            start, end = _trim_span(text, current_start, current_end)
            paragraphs.append(
                {
                    "id": len(paragraphs),
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                }
            )
            current_start = None
            current_end = None

        position += len(line)

    if current_start is not None and current_end is not None:
        start, end = _trim_span(text, current_start, current_end)
        paragraphs.append(
            {
                "id": len(paragraphs),
                "start": start,
                "end": end,
                "text": text[start:end],
            }
        )

    return paragraphs


def _make_segment(
    text: str,
    start: int,
    end: int,
    paragraph_index: int,
    sentence_index: int,
    clause_index: Optional[int],
    kind: str,
    segment_id: int,
) -> Optional[dict]:
    start, end = _trim_span(text, start, end)
    segment_text = text[start:end].strip()
    words = _word_count(segment_text)
    normalized_segment = _normalize_text(segment_text).strip(" .,;:")

    if words < MIN_SEGMENT_WORDS or normalized_segment in DISCOURSE_ONLY_PHRASES:
        return None

    return {
        "id": segment_id,
        "text": segment_text,
        "start": start,
        "end": end,
        "paragraph_index": paragraph_index,
        "sentence_index": sentence_index,
        "clause_index": clause_index,
        "kind": kind,
        "word_count": words,
        "_tokens": _tokens(segment_text),
        "_entities": _entities(segment_text),
    }


def _clause_spans(sentence_text: str) -> list[tuple[int, int]]:
    split_points = {0, len(sentence_text)}

    for match in re.finditer(r"[,;:]", sentence_text):
        split_points.add(match.end())

    for match in CLAUSE_CONNECTOR_PATTERN.finditer(sentence_text):
        split_points.add(match.start())
        split_points.add(match.end())

    points = sorted(split_points)
    spans = []
    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]
        piece = sentence_text[start:end]
        if _word_count(piece) >= MIN_SEGMENT_WORDS:
            spans.append((start, end))

    return spans


def _split_sentence(
    text: str,
    start: int,
    end: int,
    paragraph_index: int,
    sentence_index: int,
    next_segment_id: int,
) -> list[dict]:
    sentence_text = text[start:end]
    clause_spans = _clause_spans(sentence_text)
    should_split = (
        _word_count(sentence_text) >= CLAUSE_SPLIT_WORDS
        and len(clause_spans) >= 2
    )

    if not should_split:
        segment = _make_segment(
            text,
            start,
            end,
            paragraph_index,
            sentence_index,
            None,
            "sentence",
            next_segment_id,
        )
        return [segment] if segment else []

    clauses = []
    for clause_index, (clause_start_offset, clause_end_offset) in enumerate(clause_spans):
        clause_start = start + clause_start_offset
        clause_end = start + clause_end_offset
        segment = _make_segment(
            text,
            clause_start,
            clause_end,
            paragraph_index,
            sentence_index,
            clause_index,
            "clause",
            next_segment_id + len(clauses),
        )
        if segment:
            clauses.append(segment)

    if len(clauses) >= 2:
        return clauses

    segment = _make_segment(
        text,
        start,
        end,
        paragraph_index,
        sentence_index,
        None,
        "sentence",
        next_segment_id,
    )
    return [segment] if segment else []


def _split_document(text: str) -> tuple[list[dict], list[dict], bool]:
    paragraphs = _extract_paragraphs(text)
    segments = []
    truncated = False

    for paragraph in paragraphs:
        paragraph_text = paragraph["text"]
        sentence_index = 0
        for match in re.finditer(r"[^.!?;\n]+(?:[.!?;]+|$)", paragraph_text):
            sentence_start = paragraph["start"] + match.start()
            sentence_end = paragraph["start"] + match.end()
            sentence_segments = _split_sentence(
                text,
                sentence_start,
                sentence_end,
                paragraph["id"],
                sentence_index,
                len(segments),
            )
            for segment in sentence_segments:
                segment["id"] = len(segments)
                segments.append(segment)
                if len(segments) >= MAX_SEGMENTS:
                    truncated = True
                    break

            sentence_index += 1
            if truncated:
                break

        if truncated:
            break

    return segments, paragraphs, truncated


def _candidate_score(first: dict, second: dict) -> float:
    shared_tokens = first["_tokens"] & second["_tokens"]
    shared_entities = first["_entities"] & second["_entities"]
    all_tokens = first["_tokens"] | second["_tokens"]
    lexical_score = len(shared_tokens) * 3
    entity_score = len(shared_entities) * 5
    jaccard = len(shared_tokens) / len(all_tokens) if all_tokens else 0
    segment_distance = second["id"] - first["id"]
    paragraph_distance = abs(second["paragraph_index"] - first["paragraph_index"])
    proximity_score = max(0, LOCAL_WINDOW + 1 - segment_distance)

    if paragraph_distance == 0:
        proximity_score += 2
    elif paragraph_distance == 1:
        proximity_score += 1

    return lexical_score + entity_score + (jaccard * 5) + proximity_score


def _add_candidate(
    candidates: dict[tuple[int, int], dict],
    segments: list[dict],
    first_index: int,
    second_index: int,
    reason: str,
    base_score: float = 0,
) -> None:
    key = (first_index, second_index)
    score = _candidate_score(segments[first_index], segments[second_index]) + base_score

    if key not in candidates:
        candidates[key] = {"score": score, "reasons": {reason}}
    else:
        candidates[key]["score"] = max(candidates[key]["score"], score)
        candidates[key]["reasons"].add(reason)


def _select_candidate_pairs(
    segments: list[dict],
    max_pairs: Optional[int],
) -> tuple[list[dict], int]:
    total_possible = len(segments) * (len(segments) - 1) // 2
    if len(segments) < 2:
        return [], total_possible

    limit = min(max_pairs or MAX_COMPARISONS, MAX_COMPARISONS)
    candidates: dict[tuple[int, int], dict] = {}

    for index in range(len(segments)):
        for next_index in range(index + 1, min(len(segments), index + LOCAL_WINDOW + 1)):
            _add_candidate(candidates, segments, index, next_index, "proximity", 4)

    for first_index, second_index in itertools.combinations(range(len(segments)), 2):
        first = segments[first_index]
        second = segments[second_index]
        paragraph_distance = abs(first["paragraph_index"] - second["paragraph_index"])
        has_shared_topic = bool(first["_tokens"] & second["_tokens"]) or bool(
            first["_entities"] & second["_entities"]
        )

        if len(segments) <= 24:
            _add_candidate(candidates, segments, first_index, second_index, "short_document")
        elif has_shared_topic:
            _add_candidate(candidates, segments, first_index, second_index, "shared_topic", 2)
        elif paragraph_distance <= 1 and second_index - first_index <= 8:
            _add_candidate(candidates, segments, first_index, second_index, "nearby_paragraph")

    ordered = sorted(
        candidates.items(),
        key=lambda item: (
            item[1]["score"],
            -(item[0][1] - item[0][0]),
        ),
        reverse=True,
    )

    selected = []
    for (first_index, second_index), data in ordered[:limit]:
        selected.append(
            {
                "premise_index": first_index,
                "hypothesis_index": second_index,
                "candidate_score": round(data["score"], 4),
                "reasons": sorted(data["reasons"]),
            }
        )

    return selected, total_possible


def _normalize_scores(scores: dict) -> dict:
    return {_label_key(label): score for label, score in scores.items()}


def _calibrate_result(result: dict) -> dict:
    result["label"] = _label_key(result["label"])
    result["scores"] = _normalize_scores(result.get("scores", {}))
    result["confidence"] = result["scores"].get(result["label"], 0)
    result["is_contradiction"] = result["label"] == "contradiction"

    contradiction_score = result["scores"].get("contradiction", 0)
    neutral_score = result["scores"].get("neutral", 0)
    if (
        result["is_contradiction"]
        and contradiction_score < CONTRADICTION_ALERT_THRESHOLD
        and neutral_score >= contradiction_score - NEUTRAL_MARGIN
    ):
        result["label"] = "neutral"
        result["confidence"] = neutral_score
        result["is_contradiction"] = False
        result["calibration"] = "low_confidence_contradiction"

    return result


def _agent_context(premise: dict, hypothesis: dict, result: dict) -> dict:
    segment_distance = abs(hypothesis["id"] - premise["id"])
    paragraph_distance = abs(hypothesis["paragraph_index"] - premise["paragraph_index"])
    shared_tokens = premise["_tokens"] & hypothesis["_tokens"]
    shared_entities = premise["_entities"] & hypothesis["_entities"]
    entities_are_disjoint = (
        bool(premise["_entities"])
        and bool(hypothesis["_entities"])
        and not shared_entities
    )

    if segment_distance <= 2:
        distance_bucket = "near"
    elif segment_distance <= 6:
        distance_bucket = "medium"
    else:
        distance_bucket = "far"

    if paragraph_distance == 0:
        paragraph_scope = "same_paragraph"
    elif paragraph_distance == 1:
        paragraph_scope = "adjacent_paragraph"
    else:
        paragraph_scope = "distant_paragraph"

    if shared_entities:
        topic_scope = "shared_entity"
    elif shared_tokens:
        topic_scope = "shared_terms"
    elif entities_are_disjoint:
        topic_scope = "different_entities"
    else:
        topic_scope = "weak_topic"

    return {
        "distance_bucket": distance_bucket,
        "paragraph_scope": paragraph_scope,
        "topic_scope": topic_scope,
        "source": result.get("source", "model"),
        "calibration": result.get("calibration"),
        "shared_terms": sorted(shared_tokens)[:8],
        "shared_entities": sorted(shared_entities)[:8],
    }


def _apply_agent_decision(result: dict, context: Optional[dict] = None) -> dict:
    if result["is_contradiction"]:
        result["rl_decision"] = decide(result["confidence"], context)
        result["agent_alert"] = result["rl_decision"]["show_alert"]
    return result


def _topic_guard_result(result: dict, premise: dict, hypothesis: dict) -> dict:
    if result.get("source") == "cache" or not result.get("is_contradiction"):
        return result

    shared_tokens = premise["_tokens"] & hypothesis["_tokens"]
    shared_entities = premise["_entities"] & hypothesis["_entities"]
    entities_are_disjoint = (
        bool(premise["_entities"])
        and bool(hypothesis["_entities"])
        and not shared_entities
    )
    has_shared_topic = bool(shared_tokens) or bool(shared_entities)

    if entities_are_disjoint and not shared_tokens:
        result["label"] = "neutral"
        result["confidence"] = max(result["scores"].get("neutral", 0), 0.5)
        result["is_contradiction"] = False
        result["calibration"] = "different_entities"
    elif not has_shared_topic:
        result["label"] = "neutral"
        result["confidence"] = max(result["scores"].get("neutral", 0), 0.5)
        result["is_contradiction"] = False
        result["calibration"] = "no_topic_overlap"

    return result


def _analyze_text_pair(premise: str, hypothesis: str) -> dict:
    cached_label = check_cache(premise, hypothesis)
    if cached_label:
        label = _label_key(cached_label)
        result = {
            "label": label,
            "scores": {label: 1.0},
            "confidence": 1.0,
            "is_contradiction": label == "contradiction",
            "source": "cache",
        }
    else:
        memory_match = find_similar_correction(premise, hypothesis)
        if memory_match:
            label = _label_key(memory_match["label"])
            result = {
                "label": label,
                "scores": {label: memory_match["similarity"]},
                "confidence": memory_match["similarity"],
                "is_contradiction": label == "contradiction",
                "source": "memory",
                "memory_match": memory_match,
            }
        else:
            result = detect_pairs([(premise, hypothesis)], DEFAULT_BATCH_SIZE)[0]
            result = _calibrate_result(result)
            result["source"] = "model"

    return _apply_agent_decision(result)


def _build_comparison(
    pair_ref: dict,
    segments: list[dict],
    result: dict,
    context: Optional[dict] = None,
) -> dict:
    premise = segments[pair_ref["premise_index"]]
    hypothesis = segments[pair_ref["hypothesis_index"]]
    shared_tokens = sorted(premise["_tokens"] & hypothesis["_tokens"])
    shared_entities = sorted(premise["_entities"] & hypothesis["_entities"])

    return {
        **result,
        "id": f"{premise['id']}-{hypothesis['id']}",
        "premise_index": premise["id"],
        "hypothesis_index": hypothesis["id"],
        "premise": premise["text"],
        "hypothesis": hypothesis["text"],
        "premise_start": premise["start"],
        "premise_end": premise["end"],
        "hypothesis_start": hypothesis["start"],
        "hypothesis_end": hypothesis["end"],
        "premise_paragraph": premise["paragraph_index"],
        "hypothesis_paragraph": hypothesis["paragraph_index"],
        "candidate_score": pair_ref["candidate_score"],
        "reasons": pair_ref["reasons"],
        "shared_terms": shared_tokens[:8],
        "shared_entities": shared_entities[:8],
        "agent_context": context or {},
    }


def _analyze_document_pairs(pair_refs: list[dict], segments: list[dict]) -> list[dict]:
    comparisons: list[Optional[dict]] = [None] * len(pair_refs)
    pending_pairs: list[tuple[str, str]] = []
    pending_positions: list[int] = []

    for position, pair_ref in enumerate(pair_refs):
        premise = segments[pair_ref["premise_index"]]["text"]
        hypothesis = segments[pair_ref["hypothesis_index"]]["text"]
        cached_label = check_cache(premise, hypothesis)

        if cached_label:
            label = _label_key(cached_label)
            result = {
                "label": label,
                "scores": {label: 1.0},
                "confidence": 1.0,
                "is_contradiction": label == "contradiction",
                "source": "cache",
            }
            premise = segments[pair_ref["premise_index"]]
            hypothesis = segments[pair_ref["hypothesis_index"]]
            result = _topic_guard_result(result, premise, hypothesis)
            context = _agent_context(premise, hypothesis, result)
            comparisons[position] = _build_comparison(
                pair_ref,
                segments,
                _apply_agent_decision(result, context),
                context,
            )
        else:
            memory_match = find_similar_correction(premise, hypothesis)
            if memory_match:
                label = _label_key(memory_match["label"])
                result = {
                    "label": label,
                    "scores": {label: memory_match["similarity"]},
                    "confidence": memory_match["similarity"],
                    "is_contradiction": label == "contradiction",
                    "source": "memory",
                    "memory_match": memory_match,
                }
                premise_segment = segments[pair_ref["premise_index"]]
                hypothesis_segment = segments[pair_ref["hypothesis_index"]]
                result = _topic_guard_result(result, premise_segment, hypothesis_segment)
                context = _agent_context(premise_segment, hypothesis_segment, result)
                comparisons[position] = _build_comparison(
                    pair_ref,
                    segments,
                    _apply_agent_decision(result, context),
                    context,
                )
            else:
                pending_pairs.append((premise, hypothesis))
                pending_positions.append(position)

    if pending_pairs:
        model_results = detect_pairs(pending_pairs, DEFAULT_BATCH_SIZE)
        for position, result in zip(pending_positions, model_results):
            calibrated = _calibrate_result(result)
            calibrated["source"] = "model"
            pair_ref = pair_refs[position]
            premise = segments[pair_ref["premise_index"]]
            hypothesis = segments[pair_ref["hypothesis_index"]]
            calibrated = _topic_guard_result(calibrated, premise, hypothesis)
            context = _agent_context(premise, hypothesis, calibrated)
            comparisons[position] = _build_comparison(
                pair_ref,
                segments,
                _apply_agent_decision(calibrated, context),
                context,
            )

    return [comparison for comparison in comparisons if comparison is not None]


def _sort_comparisons(comparisons: list[dict]) -> list[dict]:
    return sorted(
        comparisons,
        key=lambda item: (
            item["label"] == "contradiction",
            item["confidence"],
            item["candidate_score"],
        ),
        reverse=True,
    )


def _aggregate_scores(comparisons: list[dict]) -> dict:
    labels = ("contradiction", "neutral", "entailment")
    weighted = {label: 0.0 for label in labels}
    max_by_label = {label: 0.0 for label in labels}

    for comparison in comparisons:
        label = comparison.get("label")
        confidence = comparison.get("confidence", 0)
        if label in weighted:
            weighted[label] += confidence
            max_by_label[label] = max(max_by_label[label], confidence)

    total = sum(weighted.values())
    if total <= 0:
        return {label: 0 for label in labels}

    share = {label: weighted[label] / total for label in labels}
    evidence = {
        label: (max_by_label[label] * 0.7) + (share[label] * 0.3)
        for label in labels
    }

    if max_by_label["contradiction"] >= CONTRADICTION_ALERT_THRESHOLD:
        evidence["contradiction"] = max_by_label["contradiction"]
        evidence["neutral"] = min(evidence["neutral"], 0.35)
        evidence["entailment"] = min(evidence["entailment"], 0.35)

    return {label: round(value, 4) for label, value in evidence.items()}


def _pick_document_result(comparisons: list[dict]) -> tuple[str, Optional[dict]]:
    if not comparisons:
        return "neutral", None

    ordered = _sort_comparisons(comparisons)
    contradictions = [
        comparison for comparison in ordered if comparison["label"] == "contradiction"
    ]
    if contradictions:
        return "contradiction", contradictions[0]

    entailments = [
        comparison for comparison in ordered if comparison["label"] == "entailment"
    ]
    if entailments:
        return "entailment", entailments[0]

    return "neutral", ordered[0]


def _public_segments(segments: list[dict]) -> list[dict]:
    return [
        {
            key: value
            for key, value in segment.items()
            if not key.startswith("_")
        }
        for segment in segments
    ]


def _build_highlights(conflicts: list[dict]) -> list[dict]:
    highlights_by_segment: dict[int, dict] = {}

    for conflict in _sort_comparisons(conflicts):
        for role in ("premise", "hypothesis"):
            segment_id = conflict[f"{role}_index"]
            if segment_id in highlights_by_segment:
                highlights_by_segment[segment_id]["pair_ids"].append(conflict["id"])
                continue

            highlights_by_segment[segment_id] = {
                "segment_id": segment_id,
                "start": conflict[f"{role}_start"],
                "end": conflict[f"{role}_end"],
                "paragraph_index": conflict[f"{role}_paragraph"],
                "pair_id": conflict["id"],
                "pair_ids": [conflict["id"]],
                "confidence": conflict["confidence"],
                "label": conflict["label"],
            }

    return sorted(highlights_by_segment.values(), key=lambda item: item["start"])


def _agent_summary(conflicts: list[dict], comparisons: list[dict]) -> dict:
    memory = memory_stats()
    policy = policy_stats()
    return {
        "learning_method": "q_learning_epsilon_greedy",
        "memory_items": memory["corrections"],
        "memory": memory,
        "policy": policy,
        "learned_states": policy["learned_states"],
        "reviewed_pairs": len(comparisons),
        "conflict_pairs": len(conflicts),
        "alerted_conflicts": sum(1 for conflict in conflicts if conflict.get("agent_alert")),
    }


@app.get("/")
def inicio():
    return {"mensaje": "Backend funcionando correctamente"}


@app.post("/detect")
def detect(req: ContradictionRequest):
    return _analyze_text_pair(req.premise, req.hypothesis)


@app.post("/detect-document")
def detect_document(req: DocumentRequest):
    text = req.text
    segments, paragraphs, truncated = _split_document(text)

    if len(segments) < 2:
        return {
            "label": "neutral",
            "scores": {},
            "is_contradiction": False,
            "source": "document",
            "segments": _public_segments(segments),
            "paragraphs": paragraphs,
            "comparisons": [],
            "conflicts": [],
            "highlights": [],
            "analysis_count": 0,
            "candidate_count": 0,
            "total_possible_pairs": 0,
            "truncated": truncated,
            "agent_summary": _agent_summary([], []),
            "message": "Se necesitan al menos dos enunciados para comparar coherencia.",
        }

    pair_refs, total_possible_pairs = _select_candidate_pairs(segments, req.max_pairs)
    comparisons = _analyze_document_pairs(pair_refs, segments)
    ordered_comparisons = _sort_comparisons(comparisons)
    conflicts = [
        comparison
        for comparison in ordered_comparisons
        if comparison["label"] == "contradiction"
    ][:MAX_RETURNED_CONFLICTS]

    label, strongest_pair = _pick_document_result(comparisons)
    returned_comparisons = []
    seen_ids = set()
    for comparison in conflicts + ordered_comparisons:
        if comparison["id"] in seen_ids:
            continue
        returned_comparisons.append(comparison)
        seen_ids.add(comparison["id"])
        if len(returned_comparisons) >= MAX_RETURNED_COMPARISONS:
            break

    response = {
        "label": label,
        "scores": _aggregate_scores(comparisons),
        "is_contradiction": label == "contradiction",
        "source": "document",
        "segments": _public_segments(segments),
        "paragraphs": paragraphs,
        "comparisons": returned_comparisons,
        "conflicts": conflicts,
        "highlights": _build_highlights(conflicts),
        "analysis_count": len(comparisons),
        "candidate_count": len(pair_refs),
        "total_possible_pairs": total_possible_pairs,
        "strongest_pair": strongest_pair,
        "truncated": truncated,
        "agent_summary": _agent_summary(conflicts, comparisons),
    }

    if strongest_pair and strongest_pair.get("rl_decision"):
        response["rl_decision"] = strongest_pair["rl_decision"]

    return response


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    reward = 1 if req.accepted else -1
    updated_q = update(req.confidence, req.action, reward, req.context)

    correction_saved = False
    if not req.accepted and req.correct_label and req.premise and req.hypothesis:
        save_correction(req.premise, req.hypothesis, req.correct_label)
        correction_saved = True

    total = count_corrections()

    return {
        "reward": reward,
        "updated_q_table": updated_q,
        "correction_saved": correction_saved,
        "total_corrections": total,
        "ready_to_retrain": total > 0 and total % 50 == 0,
    }
