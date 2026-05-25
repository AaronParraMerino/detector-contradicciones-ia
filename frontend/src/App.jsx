// frontend/src/App.jsx

import { useState } from 'react'
import './App.css'

const API_BASE_URL = 'http://127.0.0.1:8000'

function App() {
  const [texto, setTexto] = useState('')
  const [resultado, setResultado] = useState(null)
  const [cargando, setCargando] = useState(false)
  const [error, setError] = useState('')

  const analizarTexto = async () => {
    if (!texto.trim()) {
      setError('Primero escribe o pega un texto para analizar.')
      setResultado(null)
      return
    }

    setCargando(true)
    setError('')
    setResultado(null)

    const posiblesRutas = [
      '/analizar',
      '/analyze',
      '/detectar',
      '/contradicciones',
      '/predict',
    ]

    for (const ruta of posiblesRutas) {
      try {
        const response = await fetch(`${API_BASE_URL}${ruta}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            texto,
            text: texto,
          }),
        })

        if (response.ok) {
          const data = await response.json()
          setResultado({
            ruta,
            data,
          })
          setCargando(false)
          return
        }
      } catch {
        // Sigue probando otra ruta
      }
    }

    setError(
      'No se pudo conectar con un endpoint de análisis. Revisa en http://127.0.0.1:8000/docs cuál es la ruta correcta del backend.'
    )
    setCargando(false)
  }

  const cargarEjemplo = () => {
    setTexto(
      'El informe indica que el sistema usa SQLite para almacenar datos. Sin embargo, en la sección de arquitectura se afirma que no se utiliza ninguna base de datos. Además, se menciona que el modelo aprende con retroalimentación, pero luego se dice que el sistema no guarda correcciones del usuario.'
    )
    setResultado(null)
    setError('')
  }

  const limpiar = () => {
    setTexto('')
    setResultado(null)
    setError('')
  }

  return (
    <main className="app">
      <section className="hero-section">
        <div className="badge">Proyecto IA</div>

        <h1>Detector de Contradicciones</h1>

        <p className="subtitle">
          Analiza textos y encuentra posibles contradicciones, errores de
          coherencia o conflictos entre ideas usando Procesamiento de Lenguaje
          Natural.
        </p>
      </section>

      <section className="content-grid">
        <div className="card input-card">
          <div className="card-header">
            <div>
              <h2>Texto a analizar</h2>
              <p>Pega un informe, párrafo o contenido académico.</p>
            </div>

            <button className="secondary-button" onClick={cargarEjemplo}>
              Cargar ejemplo
            </button>
          </div>

          <textarea
            value={texto}
            onChange={(event) => setTexto(event.target.value)}
            placeholder="Ejemplo: El sistema usa SQLite, pero luego el informe indica que no se utiliza ninguna base de datos..."
          />

          <div className="actions">
            <button className="primary-button" onClick={analizarTexto}>
              {cargando ? 'Analizando...' : 'Analizar texto'}
            </button>

            <button className="ghost-button" onClick={limpiar}>
              Limpiar
            </button>
          </div>

          {error && <div className="error-box">{error}</div>}
        </div>

        <div className="card result-card">
          <h2>Resultado del análisis</h2>

          {!resultado && !cargando && (
            <div className="empty-state">
              <div className="empty-icon">🧠</div>
              <p>
                El resultado aparecerá aquí cuando el backend responda con el
                análisis del texto.
              </p>
            </div>
          )}

          {cargando && (
            <div className="loading-state">
              <div className="loader"></div>
              <p>Buscando contradicciones...</p>
            </div>
          )}

          {resultado && (
            <div className="result-box">
              <div className="success-banner">
                Conectado correctamente con: <strong>{resultado.ruta}</strong>
              </div>

              <pre>{JSON.stringify(resultado.data, null, 2)}</pre>
            </div>
          )}
        </div>
      </section>

      <section className="info-section">
        <div className="info-card">
          <span>01</span>
          <h3>Procesamiento</h3>
          <p>El sistema recibe texto y lo prepara para el análisis.</p>
        </div>

        <div className="info-card">
          <span>02</span>
          <h3>Detección</h3>
          <p>Busca frases que puedan contradecirse entre sí.</p>
        </div>

        <div className="info-card">
          <span>03</span>
          <h3>Retroalimentación</h3>
          <p>Permite mejorar el modelo con correcciones del usuario.</p>
        </div>
      </section>
    </main>
  )
}

export default App