import { useEffect, useRef, useState, useCallback } from "react"

const API = "http://localhost:8000"

const PROMPT_PRESETS = [
  {
    label: "🧍 Behavior focus",
    text:  "Describe what this person is doing and whether their behavior seems unusual. Reply with one sentence.",
  },
  {
    label: "🎒 Carried items",
    text:  "List any bags or objects this person is carrying or holding. Reply with one sentence.",
  },
  {
    label: "🚨 Threat only",
    text:  "Is anyone showing aggressive or threatening behavior? Answer yes or no and one reason.",
  },
  {
    label: "👁️ Scene overview",
    text:  "Describe exactly what is happening in this scene in one factual sentence.",
  },
]

function Toggle({ checked, onChange, disabled = false, label, sublabel, color = "blue" }) {
  const colors = {
    blue:   checked ? "bg-blue-600"   : "bg-gray-700",
    purple: checked ? "bg-purple-600" : "bg-gray-700",
    green:  checked ? "bg-green-600"  : "bg-gray-700",
  }
  return (
    <label className={`flex items-center gap-3 cursor-pointer select-none
                       ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}>
      <div
        onClick={() => !disabled && onChange(!checked)}
        className={`relative w-11 h-6 rounded-full transition-colors duration-200
                    ${colors[color]}`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full
                      shadow transition-transform duration-200
                      ${checked ? "translate-x-5" : "translate-x-0"}`}
        />
      </div>
      <div>
        <p className={`text-sm font-semibold
                       ${checked ? "text-white" : "text-gray-400"}`}>
          {label}
        </p>
        {sublabel && (
          <p className="text-xs text-gray-500 leading-tight">{sublabel}</p>
        )}
      </div>
    </label>
  )
}

function VramBar({ vram }) {
  if (!vram) return null
  const pct   = vram.usage_pct ?? 0
  const color = pct > 85 ? "bg-red-500" : pct > 65 ? "bg-yellow-400" : "bg-green-500"
  const text  = pct > 85 ? "text-red-400" : pct > 65 ? "text-yellow-400" : "text-green-400"
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">{vram.gpu_name ?? "GPU"}</span>
        <span className={`font-semibold ${text}`}>{pct}%</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-1.5">
        <div
          className={`h-1.5 rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-gray-500">
        <span>{vram.reserved_gb?.toFixed(1)}GB used</span>
        <span>{vram.total_gb?.toFixed(1)}GB total</span>
      </div>
      {pct > 85 && <p className="text-xs text-red-400">⚠️ High VRAM — disable VLM</p>}
    </div>
  )
}

const TABS = [
  { id: "settings", label: "⚙️" },
  { id: "alerts",   label: "🚨" },
  { id: "persons",  label: "👤" },
  { id: "prompt",   label: "📝" },
]

export default function App() {
  const [status,         setStatus]         = useState(null)
  const [alerts,         setAlerts]         = useState([])
  const [persons,        setPersons]        = useState([])
  const [vram,           setVram]           = useState(null)
  const [activeTab,      setActiveTab]      = useState("settings")
  const [sourceType,     setSourceType]     = useState("camera")
  const [cameraIndex,    setCameraIndex]    = useState(0)
  const [rtspPath,       setRtspPath]       = useState("")
  const [customPrompt,   setCustomPrompt]   = useState("")
  const [promptSaving,   setPromptSaving]   = useState(false)
  const [promptSaved,    setPromptSaved]    = useState(false)
  const [intervalVal,    setIntervalVal]    = useState(15)
  const [intervalSaving, setIntervalSaving] = useState(false)
  const [toggling,       setToggling]       = useState({ yolo: false, vlm: false })

  const fileRef = useRef(null)

  // ── Polling ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    const poll = async () => {
      try {
        const [s, a, p, v] = await Promise.all([
          fetch(`${API}/status`).then(r => r.json()).catch(() => null),
          fetch(`${API}/alerts`).then(r => r.json()).catch(() => []),
          fetch(`${API}/persons`).then(r => r.json()).catch(() => []),
          fetch(`${API}/vram`).then(r => r.json()).catch(() => null),
        ])
        if (s) {
          setStatus(s)
          if (!intervalSaving) setIntervalVal(Math.round(s.vlm_interval ?? 15))
          if (!promptSaving && s.custom_prompt !== undefined)
            setCustomPrompt(s.custom_prompt)
          // Clear toggling state when backend confirms change
          setToggling(prev => ({
            yolo: prev.yolo && s.mode_switching,
            vlm:  prev.vlm  && s.mode_switching,
          }))
        }
        setAlerts(Array.isArray(a) ? a.slice(-40).reverse() : [])
        setPersons(Array.isArray(p) ? p.slice(-30).reverse() : [])
        if (v && !v.error) setVram(v)
      } catch (_) {}
    }
    poll()
    const iv = setInterval(poll, 800)
    return () => clearInterval(iv)
  }, [intervalSaving, promptSaving])

  // ── Handlers ─────────────────────────────────────────────────────────────────
  const toggleYolo = useCallback(async (val) => {
    if (toggling.yolo) return
    setToggling(p => ({ ...p, yolo: true }))
    try {
      await fetch(`${API}/yolo/${val ? "enable" : "disable"}`, { method: "POST" })
    } finally {
      setToggling(p => ({ ...p, yolo: false }))
    }
  }, [toggling.yolo])

  const toggleVlm = useCallback(async (val) => {
    if (toggling.vlm) return
    setToggling(p => ({ ...p, vlm: true }))
    try {
      await fetch(`${API}/vlm/${val ? "enable" : "disable"}`, { method: "POST" })
    } finally {
      setTimeout(() => setToggling(p => ({ ...p, vlm: false })), 1500)
    }
  }, [toggling.vlm])

  const startCamera = () =>
    fetch(`${API}/start/camera?index=${cameraIndex}`, { method: "POST" }).catch(() => {})

  const startPath = () => {
    if (!rtspPath.trim()) return
    fetch(`${API}/start/path?${new URLSearchParams({ path: rtspPath })}`,
          { method: "POST" }).catch(() => {})
  }

  const startFile = async () => {
    const file = fileRef.current?.files?.[0]
    if (!file) return
    const fd = new FormData()
    fd.append("file", file)
    await fetch(`${API}/start/video`, { method: "POST", body: fd }).catch(() => {})
  }

  const stopStream = () =>
    fetch(`${API}/stop`, { method: "POST" }).catch(() => {})

  const saveInterval = async (val) => {
    setIntervalSaving(true)
    try {
      await fetch(`${API}/vlm/interval?seconds=${val}`, { method: "POST" })
    } finally { setIntervalSaving(false) }
  }

  const savePrompt = async () => {
    setPromptSaving(true)
    try {
      await fetch(`${API}/vlm/prompt?${new URLSearchParams({ prompt: customPrompt })}`,
                  { method: "POST" })
      setPromptSaved(true)
      setTimeout(() => setPromptSaved(false), 2000)
    } finally { setPromptSaving(false) }
  }

  const clearPrompt = async () => {
    setCustomPrompt("")
    await fetch(`${API}/vlm/prompt`, { method: "DELETE" }).catch(() => {})
  }

  // ── Derived ──────────────────────────────────────────────────────────────────
  const isRunning   = status?.running      ?? false
  const yoloEnabled = status?.yolo_enabled ?? false
  const vlmEnabled  = status?.vlm_enabled  ?? false
  const isSwitching = status?.mode_switching ?? false
  const alertLevel  = status?.alert        ?? "CLEAR"
  const hasWeapons  = (status?.weapon_detections?.length ?? 0) > 0
  const vramPct     = vram?.usage_pct ?? 0
  const vramText    = vramPct > 85 ? "text-red-400" : vramPct > 65 ? "text-yellow-400" : "text-green-400"

  const alertBg = {
    CLEAR:  "bg-green-950  border-green-700  text-green-300",
    YELLOW: "bg-yellow-950 border-yellow-700 text-yellow-300",
    RED:    "bg-red-950    border-red-700    text-red-300",
  }[alertLevel] ?? ""

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white overflow-hidden">

      {/* ── VLM loading overlay ─────────────────────────────────────────────── */}
      {isSwitching && (
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center
                        bg-gray-950/90 backdrop-blur-sm">
          <div className="w-12 h-12 rounded-full border-4 border-gray-700
                          border-t-purple-500 animate-spin mb-4" />
          <p className="text-white font-bold text-lg mb-1">Loading VLM onto GPU</p>
          <p className="text-gray-400 text-sm mb-4">SmolVLM2-2.2B — please wait...</p>
          <div className="w-56">
            <VramBar vram={vram} />
          </div>
        </div>
      )}

      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-4 py-2
                         bg-gray-900 border-b border-gray-800 shrink-0 gap-2 flex-wrap">
        <h1 className="text-sm font-bold tracking-wide">
          🎯 CCTV Surveillance
        </h1>

        <div className="flex items-center gap-2 flex-wrap justify-end">
          {/* VRAM */}
          {vram && (
            <span className={`text-xs px-2 py-0.5 rounded-full border
                              border-gray-700 bg-gray-800 font-semibold ${vramText}`}>
              GPU {vramPct}%
            </span>
          )}

          {/* FPS */}
          {isRunning && status?.source_fps > 0 && (
            <span className="text-xs px-2 py-0.5 rounded-full border
                             border-gray-600 bg-gray-800 text-gray-300 font-semibold">
              📹 {status.source_fps}fps
            </span>
          )}

          {/* People count */}
          {isRunning && yoloEnabled && (
            <span className="text-xs px-2 py-0.5 rounded-full border
                             border-gray-600 bg-gray-800 text-gray-300 font-semibold">
              👥 {status?.person_count ?? 0}
            </span>
          )}

          {/* Weapon badge */}
          {hasWeapons && (
            <span className="text-xs px-2 py-0.5 rounded-full border
                             border-red-500 bg-red-950 text-red-300
                             font-semibold animate-pulse">
              🔪 Weapon detected
            </span>
          )}

          {/* Alert level */}
          <span className={`text-xs px-2 py-0.5 rounded-full border font-semibold ${alertBg}`}>
            {alertLevel === "RED" ? "🔴" : alertLevel === "YELLOW" ? "🟡" : "🟢"} {alertLevel}
          </span>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">

        {/* ── Sidebar ─────────────────────────────────────────────────────────── */}
        <aside className="w-60 shrink-0 bg-gray-900 border-r border-gray-800
                          flex flex-col overflow-hidden">

          {/* Tab bar */}
          <div className="flex border-b border-gray-800 shrink-0">
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`flex-1 py-2.5 text-sm transition-colors
                  ${activeTab === t.id
                    ? "bg-gray-800 text-white border-b-2 border-blue-500"
                    : "text-gray-500 hover:text-gray-300 hover:bg-gray-800/50"}`}
              >
                {t.label}
                {t.id === "alerts"  && alerts.length  > 0 && (
                  <span className="ml-1 text-xs bg-red-600 text-white
                                   rounded-full px-1">{alerts.length}</span>
                )}
                {t.id === "persons" && persons.length > 0 && (
                  <span className="ml-1 text-xs bg-blue-600 text-white
                                   rounded-full px-1">{persons.length}</span>
                )}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-4">

            {/* ── Settings ─────────────────────────────────────────────────── */}
            {activeTab === "settings" && (
              <>
                {/* Source */}
                <section>
                  <p className="text-xs text-gray-500 font-semibold uppercase
                                tracking-wider mb-2">
                    📹 Source
                  </p>
                  <div className="flex gap-1 mb-2">
                    {["camera", "file", "path"].map(t => (
                      <button
                        key={t}
                        onClick={() => setSourceType(t)}
                        className={`flex-1 text-xs py-1 rounded font-semibold capitalize
                          ${sourceType === t
                            ? "bg-blue-700 text-white"
                            : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}
                      >
                        {t}
                      </button>
                    ))}
                  </div>

                  {sourceType === "camera" && (
                    <div className="flex gap-1.5">
                      <input
                        type="number" min={0} value={cameraIndex}
                        onChange={e => setCameraIndex(Number(e.target.value))}
                        className="w-12 text-center text-xs bg-gray-800 border
                                   border-gray-700 rounded px-1 py-1.5 text-white
                                   focus:outline-none focus:border-blue-500"
                      />
                      <button onClick={startCamera}
                              className="flex-1 text-xs py-1.5 rounded bg-blue-700
                                         hover:bg-blue-600 font-semibold">
                        Start
                      </button>
                    </div>
                  )}

                  {sourceType === "file" && (
                    <div className="flex flex-col gap-1.5">
                      <input type="file" ref={fileRef} accept="video/*"
                             className="text-xs text-gray-400
                                        file:mr-2 file:text-xs file:bg-gray-700
                                        file:border-0 file:rounded file:text-gray-200
                                        file:py-0.5 file:px-2 file:cursor-pointer" />
                      <button onClick={startFile}
                              className="text-xs py-1.5 rounded bg-blue-700
                                         hover:bg-blue-600 font-semibold">
                        Upload & Start
                      </button>
                    </div>
                  )}

                  {sourceType === "path" && (
                    <div className="flex flex-col gap-1.5">
                      <input value={rtspPath}
                             onChange={e => setRtspPath(e.target.value)}
                             placeholder="rtsp://... or /path/to/video"
                             className="text-xs bg-gray-800 border border-gray-700
                                        rounded px-2 py-1.5 text-white placeholder-gray-600
                                        focus:outline-none focus:border-blue-500" />
                      <button onClick={startPath}
                              className="text-xs py-1.5 rounded bg-blue-700
                                         hover:bg-blue-600 font-semibold">
                        Start
                      </button>
                    </div>
                  )}

                  {isRunning && (
                    <button onClick={stopStream}
                            className="w-full mt-2 text-xs py-1.5 rounded bg-red-900
                                       hover:bg-red-800 font-semibold text-red-200">
                      ⏹ Stop Stream
                    </button>
                  )}
                </section>

                {/* ── Feature toggles ──────────────────────────────────────── */}
                <section>
                  <p className="text-xs text-gray-500 font-semibold uppercase
                                tracking-wider mb-3">
                    Feature Toggles
                  </p>
                  <div className="flex flex-col gap-4">
                    <Toggle
                      checked={yoloEnabled}
                      onChange={toggleYolo}
                      disabled={toggling.yolo || !isRunning}
                      label="YOLO Detection"
                      sublabel="Person tracking + weapon alerts"
                      color="blue"
                    />
                    <Toggle
                      checked={vlmEnabled}
                      onChange={toggleVlm}
                      disabled={toggling.vlm || isSwitching || !isRunning}
                      label="VLM Analysis"
                      sublabel="Scene & person descriptions"
                      color="purple"
                    />
                  </div>
                  {!isRunning && (
                    <p className="text-xs text-gray-600 mt-2 text-center">
                      Start a stream to enable toggles
                    </p>
                  )}
                </section>

                {/* VLM interval — only when VLM on */}
                {vlmEnabled && (
                  <section>
                    <p className="text-xs text-gray-500 font-semibold uppercase
                                  tracking-wider mb-2">
                      🧠 VLM Interval
                    </p>
                    <div className="flex items-center gap-2">
                      <input
                        type="range" min={5} max={120} step={5}
                        value={intervalVal}
                        onChange={e  => setIntervalVal(Number(e.target.value))}
                        onMouseUp={e  => saveInterval(Number(e.target.value))}
                        onTouchEnd={() => saveInterval(intervalVal)}
                        className="flex-1 accent-purple-500 cursor-pointer"
                      />
                      <span className="text-xs text-gray-300 w-8 text-right font-mono">
                        {intervalVal}s
                      </span>
                    </div>
                  </section>
                )}

                {/* VRAM */}
                <section className="mt-auto">
                  <p className="text-xs text-gray-500 font-semibold uppercase
                                tracking-wider mb-2">
                    ⚡ GPU Memory
                  </p>
                  <div className="bg-gray-800 rounded p-2.5 border border-gray-700">
                    {vram
                      ? <VramBar vram={vram} />
                      : <p className="text-xs text-gray-600">Backend not connected</p>
                    }
                  </div>
                </section>
              </>
            )}

            {/* ── Alerts ───────────────────────────────────────────────────── */}
            {activeTab === "alerts" && (
              <div className="flex flex-col gap-2">
                {alerts.length === 0 ? (
                  <div className="flex flex-col items-center py-10 gap-2 text-gray-600">
                    <span className="text-3xl">🔕</span>
                    <p className="text-xs">No alerts yet</p>
                  </div>
                ) : alerts.map((a, i) => (
                  <div key={i}
                       className={`rounded px-2.5 py-2 border text-xs
                         ${a.alert === "RED"
                           ? "bg-red-950    border-red-800    text-red-300"
                           : a.alert === "YELLOW"
                           ? "bg-yellow-950 border-yellow-800 text-yellow-300"
                           : "bg-gray-800   border-gray-700   text-gray-400"}`}>
                    <div className="flex justify-between mb-0.5">
                      <span className="font-bold">{a.alert}</span>
                      <span className="text-gray-500">{a.time}</span>
                    </div>
                    <p>{a.reason}</p>
                    {a.vlm && (
                      <p className="mt-1 italic text-gray-400 border-t
                                    border-gray-700/50 pt-1">
                        💬 {a.vlm}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* ── Persons ──────────────────────────────────────────────────── */}
            {activeTab === "persons" && (
              <div className="flex flex-col gap-2">
                {persons.length === 0 ? (
                  <div className="flex flex-col items-center py-10 gap-2 text-gray-600">
                    <span className="text-3xl">👤</span>
                    <p className="text-xs">No persons logged yet</p>
                    {!vlmEnabled && (
                      <p className="text-xs text-center">
                        Enable VLM to get person descriptions
                      </p>
                    )}
                  </div>
                ) : persons.map((p, i) => (
                  <div key={i}
                       className="rounded px-2.5 py-2 bg-gray-800
                                  border border-gray-700 text-xs">
                    <div className="flex justify-between mb-0.5">
                      <span className="font-bold text-blue-300">ID #{p.track_id}</span>
                      <span className="text-gray-500">{p.time}</span>
                    </div>
                    <p className="text-gray-300 leading-snug">{p.description}</p>
                  </div>
                ))}
              </div>
            )}

            {/* ── Prompt ───────────────────────────────────────────────────── */}
            {activeTab === "prompt" && (
              <div className="flex flex-col gap-3">
                <div className="bg-gray-800 rounded p-2.5 border border-gray-700
                                text-xs text-gray-400">
                  Overrides the default VLM prompts.
                  Leave blank to use built-in defaults.
                </div>

                <textarea
                  value={customPrompt}
                  onChange={e => setCustomPrompt(e.target.value)}
                  placeholder={"Describe this person's actions.\nReply with one sentence."}
                  rows={6}
                  className="w-full bg-gray-800 border border-gray-700 rounded
                             px-2.5 py-2 text-xs text-white placeholder-gray-600
                             resize-none focus:outline-none focus:border-purple-500"
                />

                <div className="flex gap-2">
                  <button
                    onClick={savePrompt}
                    disabled={promptSaving}
                    className={`flex-1 text-xs py-1.5 rounded font-semibold transition-all
                      ${promptSaved
                        ? "bg-green-700 text-white"
                        : "bg-purple-700 hover:bg-purple-600 text-white"
                      } disabled:opacity-50`}
                  >
                    {promptSaved ? "✅ Saved!" : "💾 Save"}
                  </button>
                  {customPrompt && (
                    <button onClick={clearPrompt}
                            className="px-3 text-xs py-1.5 rounded bg-gray-700
                                       hover:bg-gray-600 text-gray-300 font-semibold">
                      Clear
                    </button>
                  )}
                </div>

                <div className="bg-gray-800 rounded p-2.5 border border-gray-700">
                  <p className="text-xs text-gray-400 font-semibold mb-1.5">
                    💡 Tips
                  </p>
                  <ul className="text-xs text-gray-500 space-y-1">
                    <li>• End with "Reply with only one sentence."</li>
                    <li>• Shorter = faster inference</li>
                    <li>• Applies to all VLM calls</li>
                  </ul>
                </div>

                <div>
                  <p className="text-xs text-gray-500 font-semibold uppercase
                                tracking-wider mb-1.5">
                    Quick Presets
                  </p>
                  {PROMPT_PRESETS.map((p, i) => (
                    <button key={i} onClick={() => setCustomPrompt(p.text)}
                            className="w-full text-left text-xs px-2.5 py-1.5 rounded mb-1
                                       bg-gray-800 border border-gray-700 text-gray-300
                                       hover:bg-gray-700 transition-colors">
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>
            )}

          </div>
        </aside>

        {/* ── Main video ──────────────────────────────────────────────────────── */}
        <main className="flex-1 flex flex-col overflow-hidden min-w-0">

          {/* Alert banner */}
          {alertLevel !== "CLEAR" && (
            <div className={`px-4 py-1.5 text-sm font-semibold flex items-center
                             gap-3 shrink-0 border-b
              ${alertLevel === "RED"
                ? "bg-red-900/60    text-red-200    border-red-800"
                : "bg-yellow-900/60 text-yellow-200 border-yellow-800"}`}>
              <span>{alertLevel === "RED" ? "🔴" : "🟡"}</span>
              <span className="truncate">{status?.reason}</span>
              {status?.vlm_description && (
                <span className="text-xs opacity-70 hidden lg:inline truncate">
                  — {status.vlm_description}
                </span>
              )}
            </div>
          )}

          {/* Video */}
          <div className="flex-1 relative flex items-center justify-center
                          bg-black overflow-hidden">
            {isRunning ? (
              <>
                <img
                  src={`${API}/video_feed`}
                  className="max-h-full max-w-full object-contain"
                  alt="Live Feed"
                />

                {/* Status pills — top left */}
                <div className="absolute top-2 left-2 flex gap-1.5 pointer-events-none">
                  <span className={`text-xs px-2 py-0.5 rounded-full font-semibold
                                    border backdrop-blur-sm bg-black/50
                    ${yoloEnabled
                      ? "text-blue-300   border-blue-700"
                      : "text-gray-500   border-gray-700"}`}>
                    {yoloEnabled ? "🎯 YOLO" : "YOLO off"}
                  </span>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-semibold
                                    border backdrop-blur-sm bg-black/50
                    ${vlmEnabled
                      ? "text-purple-300 border-purple-700"
                      : "text-gray-500   border-gray-700"}`}>
                    {vlmEnabled ? "🧠 VLM" : "VLM off"}
                  </span>
                </div>

                {/* Bottom info */}
                <div className="absolute bottom-2 left-2 right-2 flex flex-col gap-1
                                pointer-events-none">
                  {status?.detection_summary && yoloEnabled && (
                    <span className="text-xs px-2 py-0.5 rounded bg-black/70
                                     text-green-300 self-start max-w-full truncate">
                      {status.detection_summary}
                    </span>
                  )}
                  {status?.scene_description && vlmEnabled && (
                    <span className="text-xs px-2 py-0.5 rounded bg-black/70
                                     text-blue-200 self-start max-w-full">
                      💬 {status.scene_description}
                    </span>
                  )}
                </div>
              </>
            ) : (
              <div className="flex flex-col items-center gap-3 text-gray-700">
                <span className="text-7xl">📷</span>
                <p className="text-sm font-semibold">No stream active</p>
                <p className="text-xs text-gray-600">
                  Select a source in ⚙️ and press Start
                </p>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
