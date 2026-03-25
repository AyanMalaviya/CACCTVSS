import { useEffect, useRef, useState, useCallback } from "react"

const API = "http://localhost:8000"

const PRESETS = {
  proximity: [
    "Two people are very close. Describe what they are doing. Start with: Safe / Suspicious / Threatening.",
    "Describe the interaction between these two people in one sentence. Are they arguing or fighting?",
    "Is there a physical altercation happening? Answer in one sentence.",
  ],
  count_change: [
    "Describe what is currently happening in this scene in one sentence.",
    "A person entered or left the frame. Describe what you now see in one sentence.",
    "How many people are visible and what are they doing? Reply in one sentence.",
  ],
  weapon: [
    "Is someone holding or using a bladed weapon? Describe in one sentence.",
    "What is the person doing with the detected object? Is it threatening? One sentence.",
    "Describe the dangerous object and what the person is doing with it. One sentence.",
  ],
}

const TRIGGER_META = {
  proximity:    { icon: "🤝", label: "Proximity Trigger",  sublabel: "2 persons close for 2.5s",        color: "yellow" },
  count_change: { icon: "👥", label: "Count Change",        sublabel: "Person count increases/decreases", color: "blue"   },
  weapon:       { icon: "🔪", label: "Weapon Trigger",      sublabel: "Knife / axe / scissor detected",  color: "red"    },
}

// ── Primitives ────────────────────────────────────────────────────────────────
function Toggle({ checked, onChange, disabled = false, label, sublabel, color = "blue" }) {
  const bg = color === "purple"
    ? (checked ? "bg-purple-600" : "bg-gray-700")
    : (checked ? "bg-blue-600"   : "bg-gray-700")
  return (
    <label className={`flex items-center gap-3 select-none
                       ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}>
      <div onClick={() => !disabled && onChange(!checked)}
           className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${bg}`}>
        <span className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow
                          transition-transform duration-200
                          ${checked ? "translate-x-5" : "translate-x-0"}`} />
      </div>
      <div>
        <p className={`text-sm font-semibold ${checked ? "text-white" : "text-gray-400"}`}>
          {label}
        </p>
        {sublabel && <p className="text-xs text-gray-500 leading-tight">{sublabel}</p>}
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
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between text-xs">
        <span className="text-gray-400 truncate max-w-[170px]">{vram.gpu_name ?? "GPU"}</span>
        <span className={`font-bold ${text}`}>{pct}%</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2">
        <div className={`h-2 rounded-full transition-all duration-700 ${color}`}
             style={{ width: `${Math.min(pct, 100)}%` }} />
      </div>
      <div className="flex justify-between text-xs text-gray-500">
        <span>{vram.reserved_gb?.toFixed(1)} GB used</span>
        <span>{vram.total_gb?.toFixed(1)} GB total</span>
      </div>
      {pct > 85 && (
        <p className="text-xs text-red-400 font-semibold">⚠️ Disable VLM to free VRAM</p>
      )}
    </div>
  )
}

function TriggerPromptEditor({ triggerType, value, onChange, onSave, onClear, saving, saved }) {
  const [open, setOpen] = useState(false)
  const meta = TRIGGER_META[triggerType]
  const borderColor = {
    yellow: "border-yellow-800",
    blue:   "border-blue-800",
    red:    "border-red-900",
  }[meta.color]
  const badgeColor = {
    yellow: "bg-yellow-900/60 text-yellow-300",
    blue:   "bg-blue-900/60   text-blue-300",
    red:    "bg-red-900/60    text-red-300",
  }[meta.color]

  return (
    <div className={`rounded-xl border ${borderColor} bg-gray-800/40 overflow-hidden`}>
      <div className="flex items-center gap-2.5 px-3 py-2.5 bg-gray-800/80
                      border-b border-gray-700/50">
        <span className="text-base">{meta.icon}</span>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-bold text-white leading-tight">{meta.label}</p>
          <p className="text-xs text-gray-500 truncate leading-tight">{meta.sublabel}</p>
        </div>
        {value && (
          <span className={`text-xs px-1.5 py-0.5 rounded-full font-semibold shrink-0 ${badgeColor}`}>
            custom
          </span>
        )}
      </div>

      <div className="p-3 flex flex-col gap-2">
        <textarea
          value={value}
          onChange={e => onChange(e.target.value)}
          placeholder={`Default: "${PRESETS[triggerType][0].slice(0, 48)}…"`}
          rows={3}
          className="w-full bg-gray-900 border border-gray-700 rounded-lg px-2.5 py-2
                     text-xs text-white placeholder-gray-600 resize-none
                     focus:outline-none focus:border-blue-500 transition-colors"
        />

        <div className="flex gap-1.5">
          <button
            onClick={onSave}
            disabled={saving}
            className={`flex-1 text-xs py-1.5 rounded-lg font-semibold transition-all
              ${saved
                ? "bg-green-700 text-white"
                : "bg-blue-700 hover:bg-blue-600 text-white"}
              disabled:opacity-50`}
          >
            {saved ? "✅ Saved!" : "💾 Save"}
          </button>
          {value && (
            <button
              onClick={onClear}
              className="px-3 text-xs py-1.5 rounded-lg bg-gray-700
                         hover:bg-gray-600 text-gray-300 font-semibold"
            >
              ✕
            </button>
          )}
        </div>

        <button
          onClick={() => setOpen(o => !o)}
          className="flex items-center gap-1.5 text-xs text-gray-600
                     hover:text-gray-400 transition-colors select-none"
        >
          <span className={`transition-transform duration-200 ${open ? "rotate-90" : ""}`}>
            ▶
          </span>
          Quick presets
        </button>

        {open && (
          <div className="flex flex-col gap-1">
            {PRESETS[triggerType].map((p, i) => (
              <button
                key={i}
                onClick={() => { onChange(p); setOpen(false) }}
                className="w-full text-left text-xs px-2.5 py-2 rounded-lg bg-gray-900
                           border border-gray-700/50 text-gray-500
                           hover:bg-gray-700/80 hover:text-gray-200 transition-colors"
              >
                {p.slice(0, 62)}…
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Right panel components ────────────────────────────────────────────────────
function AlertStatusCard({ status }) {
  const level = status?.alert ?? "CLEAR"
  const cfg = {
    CLEAR: {
      bg:    "bg-gradient-to-br from-green-950  to-gray-900",
      border:"border-green-900",
      text:  "text-green-200",
      sub:   "text-green-600",
      badge: "bg-green-900/80 text-green-200 border-green-800",
      dot:   "bg-green-400",
    },
    YELLOW: {
      bg:    "bg-gradient-to-br from-yellow-950 to-gray-900",
      border:"border-yellow-800",
      text:  "text-yellow-200",
      sub:   "text-yellow-600",
      badge: "bg-yellow-900/80 text-yellow-100 border-yellow-700",
      dot:   "bg-yellow-400 animate-pulse",
    },
    RED: {
      bg:    "bg-gradient-to-br from-red-950    to-gray-900",
      border:"border-red-800",
      text:  "text-red-200",
      sub:   "text-red-500",
      badge: "bg-red-900/80 text-red-100 border-red-700",
      dot:   "bg-red-400 animate-pulse",
    },
  }[level] ?? {}

  return (
    <div className={`${cfg.bg} border-b ${cfg.border} p-4`}>
      <div className="flex items-center gap-2.5 mb-3">
        <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${cfg.dot}`} />
        <span className={`text-sm font-black tracking-widest px-3 py-1
                          rounded-full border ${cfg.badge}`}>
          {level}
        </span>
        {status?.threat_type && status.threat_type !== "none" && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-gray-800/80
                           text-gray-400 font-semibold uppercase tracking-wide
                           border border-gray-700">
            {status.threat_type}
          </span>
        )}
      </div>

      {status?.reason
        ? <p className={`text-sm font-semibold leading-snug ${cfg.text}`}>
            {status.reason}
          </p>
        : <p className={`text-xs ${cfg.sub}`}>
            No threats detected — monitoring active
          </p>
      }

      {status?.description && level !== "CLEAR" && (
        <div className="mt-3 pt-3 border-t border-gray-800/80">
          <p className="text-xs text-gray-500 mb-1 font-semibold uppercase tracking-wider">
            VLM Verdict
          </p>
          <p className="text-xs text-gray-300 leading-relaxed italic">
            "{status.description}"
          </p>
        </div>
      )}
    </div>
  )
}

function SceneDescCard({ status }) {
  const desc  = status?.scene_description
  const vlmOn = status?.vlm_enabled

  return (
    <div className="p-4 border-b border-gray-800">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-purple-400 uppercase tracking-wider">
            🧠 VLM Scene
          </span>
          {vlmOn && (
            <span className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-pulse" />
          )}
        </div>
        {vlmOn && status?.vlm_interval != null && (
          <span className="text-xs text-gray-600 bg-gray-800 px-2 py-0.5
                           rounded-full border border-gray-700">
            ⏱ {status.vlm_interval}s
          </span>
        )}
      </div>

      {vlmOn && desc ? (
        <div className="bg-purple-950/25 border border-purple-900/50 rounded-xl p-3">
          <p className="text-sm text-gray-200 leading-relaxed">{desc}</p>
        </div>
      ) : vlmOn ? (
        <div className="bg-gray-800/40 border border-gray-700/50 rounded-xl p-3
                        flex items-center justify-center">
          <p className="text-xs text-gray-600 italic py-2">
            Waiting for first analysis…
          </p>
        </div>
      ) : (
        <div className="bg-gray-800/20 border border-gray-700/30 rounded-xl p-3">
          <p className="text-xs text-gray-600 text-center py-1">
            Enable VLM to see scene descriptions
          </p>
        </div>
      )}
    </div>
  )
}

function LiveStatsCard({ status, isRunning }) {
  const stats = [
    {
      icon: "👥", label: "Persons",
      value: status?.yolo_enabled ? String(status?.person_count ?? 0) : "—",
      color: "text-blue-300",
    },
    {
      icon: "📹", label: "FPS",
      value: isRunning && (status?.source_fps ?? 0) > 0
        ? String(status.source_fps) : "—",
      color: "text-gray-300",
    },
    {
      icon: "🎯", label: "YOLO",
      value: status?.yolo_enabled ? "ON" : "OFF",
      color: status?.yolo_enabled ? "text-blue-400" : "text-gray-600",
    },
    {
      icon: "🧠", label: "VLM",
      value: status?.vlm_enabled ? "ON" : "OFF",
      color: status?.vlm_enabled ? "text-purple-400" : "text-gray-600",
    },
  ]

  return (
    <div className="p-4 border-b border-gray-800">
      <p className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3">
        📊 Live Stats
      </p>
      <div className="grid grid-cols-2 gap-2 mb-3">
        {stats.map(s => (
          <div key={s.label}
               className="bg-gray-800/60 rounded-xl p-3 border border-gray-700/50">
            <p className="text-xs text-gray-500 mb-1.5">{s.icon} {s.label}</p>
            <p className={`text-xl font-black leading-none ${s.color}`}>{s.value}</p>
          </div>
        ))}
      </div>

      {status?.detection_summary && status?.yolo_enabled && (
        <div className="bg-gray-800/40 rounded-lg px-3 py-2 border border-gray-700/40">
          <p className="text-xs text-gray-400 leading-snug">
            {status.detection_summary}
          </p>
        </div>
      )}
    </div>
  )
}

function WeaponsCard({ detections }) {
  if (!detections?.length) return null

  const colorMap = {
    knife:   { bar: "bg-red-500",    text: "text-red-200",    bg: "bg-red-950/50    border-red-900"    },
    axe:     { bar: "bg-orange-500", text: "text-orange-200", bg: "bg-orange-950/50 border-orange-900" },
    scissors:{ bar: "bg-pink-500",   text: "text-pink-200",   bg: "bg-pink-950/50   border-pink-900"   },
    crowbar: { bar: "bg-teal-500",   text: "text-teal-200",   bg: "bg-teal-950/50   border-teal-900"   },
  }

  return (
    <div className="p-4 border-b border-gray-800 bg-red-950/10">
      <p className="text-xs font-bold text-red-400 uppercase tracking-wider
                    mb-3 animate-pulse">
        🔪 Weapons Detected
      </p>
      <div className="flex flex-col gap-2">
        {detections.map((d, i) => {
          const c = colorMap[d.label] ?? colorMap.knife
          return (
            <div key={i}
                 className={`rounded-xl p-3 border ${c.bg} flex items-center gap-3`}>
              <div className="flex-1 min-w-0">
                <p className={`text-sm font-black capitalize leading-none mb-2 ${c.text}`}>
                  {d.label}
                </p>
                <div className="flex items-center gap-2">
                  <div className="flex-1 bg-gray-700 rounded-full h-1.5">
                    <div className={`h-1.5 rounded-full transition-all ${c.bar}`}
                         style={{ width: `${d.confidence * 100}%` }} />
                  </div>
                  <span className={`text-xs font-bold w-9 text-right ${c.text}`}>
                    {Math.round(d.confidence * 100)}%
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function VramSection({ vram }) {
  return (
    <div className="p-4 border-b border-gray-800">
      <p className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3">
        ⚡ GPU Memory
      </p>
      {vram
        ? <VramBar vram={vram} />
        : <p className="text-xs text-gray-600 italic">GPU info unavailable</p>
      }
    </div>
  )
}

function AlertsList({ alerts }) {
  if (!alerts.length) return (
    <div className="flex flex-col items-center justify-center py-16 gap-3">
      <span className="text-4xl">🔕</span>
      <p className="text-sm font-semibold text-gray-600">No alerts yet</p>
      <p className="text-xs text-center text-gray-700 max-w-[180px]">
        Alerts appear here when YOLO or VLM detects a threat
      </p>
    </div>
  )
  return (
    <div className="p-3 flex flex-col gap-2">
      {alerts.map((a, i) => (
        <div key={i}
             className={`rounded-xl px-3 py-2.5 border text-xs
               ${a.alert === "RED"
                 ? "bg-red-950/60    border-red-900    text-red-300"
                 : a.alert === "YELLOW"
                 ? "bg-yellow-950/60 border-yellow-900 text-yellow-300"
                 : "bg-gray-800/60   border-gray-700   text-gray-400"}`}>
          <div className="flex items-center justify-between mb-1.5">
            <span className={`font-bold text-xs px-2 py-0.5 rounded-full
              ${a.alert === "RED"    ? "bg-red-900    text-red-200"
              : a.alert === "YELLOW" ? "bg-yellow-900 text-yellow-200"
              :                        "bg-gray-700   text-gray-300"}`}>
              {a.alert === "RED" ? "🔴" : a.alert === "YELLOW" ? "🟡" : "🟢"} {a.alert}
            </span>
            <span className="text-gray-600 font-mono text-xs">{a.time}</span>
          </div>
          <p className="leading-snug mb-1">{a.reason}</p>
          {a.vlm && (
            <p className="text-xs italic text-gray-500 border-t border-gray-700/50
                          pt-1.5 mt-1 leading-snug">
              💬 {a.vlm}
            </p>
          )}
        </div>
      ))}
    </div>
  )
}

function PersonsList({ persons, vlmEnabled }) {
  if (!persons.length) return (
    <div className="flex flex-col items-center justify-center py-16 gap-3">
      <span className="text-4xl">👤</span>
      <p className="text-sm font-semibold text-gray-600">No persons logged</p>
      {!vlmEnabled && (
        <p className="text-xs text-center text-gray-700 max-w-[180px]">
          Enable both YOLO + VLM to log person descriptions
        </p>
      )}
    </div>
  )
  return (
    <div className="p-3 flex flex-col gap-2">
      {persons.map((p, i) => (
        <div key={i}
             className="rounded-xl px-3 py-2.5 bg-gray-800/60
                        border border-gray-700/60">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs font-bold text-blue-400 bg-blue-950/50
                             border border-blue-900 px-2 py-0.5 rounded-full">
              ID #{p.track_id}
            </span>
            <span className="text-xs text-gray-600 font-mono">{p.time}</span>
          </div>
          <p className="text-xs text-gray-300 leading-relaxed">{p.description}</p>
        </div>
      ))}
    </div>
  )
}

// ── Right panel container ─────────────────────────────────────────────────────
function RightPanel({ status, alerts, persons, vram, isRunning }) {
  const [tab, setTab] = useState("status")
  const TABS = [
    { id: "status",  icon: "📊", label: "Status"                              },
    { id: "alerts",  icon: "🚨", label: "Alerts",  count: alerts.length       },
    { id: "persons", icon: "👤", label: "Persons", count: persons.length      },
  ]

  return (
    <aside className="w-72 shrink-0 bg-gray-900 border-l border-gray-800
                      flex flex-col overflow-hidden">
      {/* Tab bar */}
      <div className="flex border-b border-gray-800 shrink-0">
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
                  className={`flex-1 py-2.5 text-xs font-semibold transition-colors relative
                    ${tab === t.id
                      ? "bg-gray-800 text-white border-b-2 border-blue-500"
                      : "text-gray-500 hover:text-gray-300 hover:bg-gray-800/50"}`}>
            {t.icon} {t.label}
            {t.count > 0 && (
              <span className={`absolute top-1 right-1 min-w-4 h-4 px-0.5
                                text-[10px] flex items-center justify-center
                                rounded-full text-white font-bold
                                ${t.id === "alerts" ? "bg-red-600" : "bg-blue-600"}`}>
                {t.count > 9 ? "9+" : t.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-track-gray-900
                      scrollbar-thumb-gray-700">
        {tab === "status" && (
          <>
            <AlertStatusCard status={status} />
            <SceneDescCard   status={status} />
            <LiveStatsCard   status={status} isRunning={isRunning} />
            <WeaponsCard     detections={status?.weapon_detections} />
            <VramSection     vram={vram} />
          </>
        )}
        {tab === "alerts"  && <AlertsList  alerts={alerts} />}
        {tab === "persons" && <PersonsList persons={persons} vlmEnabled={status?.vlm_enabled} />}
      </div>
    </aside>
  )
}

// ── Left sidebar tabs ─────────────────────────────────────────────────────────
const LEFT_TABS = [
  { id: "settings", icon: "⚙️",  label: "Settings" },
  { id: "triggers", icon: "🎯",  label: "Triggers" },
]

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [status,         setStatus]         = useState(null)
  const [alerts,         setAlerts]         = useState([])
  const [persons,        setPersons]        = useState([])
  const [vram,           setVram]           = useState(null)
  const [activeTab,      setActiveTab]      = useState("settings")
  const [sourceType,     setSourceType]     = useState("camera")
  const [cameraIndex,    setCameraIndex]    = useState(0)
  const [rtspPath,       setRtspPath]       = useState("")
  const [toggling,       setToggling]       = useState({ yolo: false, vlm: false })
  const [intervalInput,  setIntervalInput]  = useState(10)
  const [intervalSaving, setIntervalSaving] = useState(false)
  const [prompts,        setPrompts]        = useState({ proximity: "", count_change: "", weapon: "" })
  const [promptSaving,   setPromptSaving]   = useState({ proximity: false, count_change: false, weapon: false })
  const [promptSaved,    setPromptSaved]    = useState({ proximity: false, count_change: false, weapon: false })

  const fileRef = useRef(null)

  // ── Polling ────────────────────────────────────────────────────────────────
  useEffect(() => {
    const poll = async () => {
      try {
        const [s, a, p, v] = await Promise.all([
          fetch(`${API}/status`).then(r  => r.json()).catch(() => null),
          fetch(`${API}/alerts`).then(r  => r.json()).catch(() => []),
          fetch(`${API}/persons`).then(r => r.json()).catch(() => []),
          fetch(`${API}/vram`).then(r    => r.json()).catch(() => null),
        ])
        if (s) {
          setStatus(s)
          if (!intervalSaving)
            setIntervalInput(Math.max(2, Math.min(30, Math.round(s.vlm_interval ?? 10))))
          if (s.trigger_prompts) {
            setPrompts(prev => ({
              proximity:    promptSaving.proximity    ? prev.proximity    : (s.trigger_prompts.proximity    ?? ""),
              count_change: promptSaving.count_change ? prev.count_change : (s.trigger_prompts.count_change ?? ""),
              weapon:       promptSaving.weapon       ? prev.weapon       : (s.trigger_prompts.weapon       ?? ""),
            }))
          }
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
  }, [intervalSaving, promptSaving.proximity, promptSaving.count_change, promptSaving.weapon])

  // ── Handlers ──────────────────────────────────────────────────────────────
  const toggleYolo = useCallback(async val => {
    if (toggling.yolo) return
    setToggling(p => ({ ...p, yolo: true }))
    try { await fetch(`${API}/yolo/${val ? "enable" : "disable"}`, { method: "POST" }) }
    finally { setToggling(p => ({ ...p, yolo: false })) }
  }, [toggling.yolo])

  const toggleVlm = useCallback(async val => {
    if (toggling.vlm) return
    setToggling(p => ({ ...p, vlm: true }))
    try { await fetch(`${API}/vlm/${val ? "enable" : "disable"}`, { method: "POST" }) }
    finally { setTimeout(() => setToggling(p => ({ ...p, vlm: false })), 1500) }
  }, [toggling.vlm])

  const saveInterval = async () => {
    const v = Math.max(2, Math.min(30, Number(intervalInput)))
    setIntervalInput(v)
    setIntervalSaving(true)
    try { await fetch(`${API}/vlm/interval?seconds=${v}`, { method: "POST" }) }
    finally { setIntervalSaving(false) }
  }

  const savePrompt = async type => {
    setPromptSaving(p => ({ ...p, [type]: true }))
    try {
      await fetch(
        `${API}/trigger_prompts/${type}?${new URLSearchParams({ prompt: prompts[type] })}`,
        { method: "POST" }
      )
      setPromptSaved(p => ({ ...p, [type]: true }))
      setTimeout(() => setPromptSaved(p => ({ ...p, [type]: false })), 2000)
    } finally { setPromptSaving(p => ({ ...p, [type]: false })) }
  }

  const clearPrompt = async type => {
    setPrompts(p => ({ ...p, [type]: "" }))
    await fetch(`${API}/trigger_prompts/${type}`, { method: "DELETE" }).catch(() => {})
  }

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

  // ── Derived ────────────────────────────────────────────────────────────────
  const isRunning   = status?.running        ?? false
  const yoloEnabled = status?.yolo_enabled   ?? false
  const vlmEnabled  = status?.vlm_enabled    ?? false
  const isSwitching = status?.mode_switching ?? false
  const alertLevel  = status?.alert          ?? "CLEAR"
  const hasWeapons  = (status?.weapon_detections?.length ?? 0) > 0
  const vramPct     = vram?.usage_pct ?? 0
  const vramText    = vramPct > 85 ? "text-red-400"
                    : vramPct > 65 ? "text-yellow-400"
                    : "text-green-400"

  const alertBarCfg = {
    CLEAR:  { bar: "bg-green-950/60  border-b border-green-900",  text: "text-green-300"  },
    YELLOW: { bar: "bg-yellow-950/60 border-b border-yellow-800", text: "text-yellow-200" },
    RED:    { bar: "bg-red-950/60    border-b border-red-800",    text: "text-red-200"    },
  }[alertLevel] ?? {}

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white overflow-hidden">

      {/* VLM loading overlay */}
      {isSwitching && (
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center
                        bg-gray-950/95 backdrop-blur-sm">
          <div className="w-14 h-14 rounded-full border-4 border-gray-700
                          border-t-purple-500 animate-spin mb-5" />
          <p className="text-white font-bold text-xl mb-1">Loading VLM onto GPU</p>
          <p className="text-gray-400 text-sm mb-6">SmolVLM2-2.2B — please wait…</p>
          <div className="w-64 bg-gray-900 rounded-2xl p-4 border border-gray-800">
            <VramBar vram={vram} />
          </div>
        </div>
      )}

      {/* ── Top header ──────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-5 py-2.5
                         bg-gray-900 border-b border-gray-800 shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-lg">🎯</span>
          <div>
            <h1 className="text-sm font-black tracking-wide leading-none">
              CCTV Surveillance
            </h1>
            <p className="text-xs text-gray-600 leading-tight">
              YOLO26n + SmolVLM2
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {vram && (
            <div className="flex items-center gap-1.5 bg-gray-800 rounded-full
                            px-3 py-1 border border-gray-700">
              <span className={`w-1.5 h-1.5 rounded-full ${
                vramPct > 85 ? "bg-red-400 animate-pulse"
                : vramPct > 65 ? "bg-yellow-400" : "bg-green-400"
              }`} />
              <span className={`text-xs font-bold ${vramText}`}>
                GPU {vramPct}%
              </span>
              <span className="text-xs text-gray-600">
                {vram.free_gb?.toFixed(1)}GB free
              </span>
            </div>
          )}

          {isRunning && (status?.source_fps ?? 0) > 0 && (
            <div className="bg-gray-800 rounded-full px-3 py-1 border border-gray-700">
              <span className="text-xs text-gray-300 font-semibold">
                📹 {status.source_fps}fps
              </span>
            </div>
          )}

          {hasWeapons && (
            <div className="bg-red-900/80 rounded-full px-3 py-1 border
                            border-red-700 animate-pulse">
              <span className="text-xs text-red-200 font-bold">
                🔪 Weapon Detected
              </span>
            </div>
          )}

          <div className={`rounded-full px-3 py-1 border text-xs font-bold
                           ${alertLevel === "RED"
                             ? "bg-red-900/80    border-red-700    text-red-200"
                             : alertLevel === "YELLOW"
                             ? "bg-yellow-900/80 border-yellow-700 text-yellow-200"
                             : "bg-green-900/80  border-green-800  text-green-300"}`}>
            {alertLevel === "RED" ? "🔴" : alertLevel === "YELLOW" ? "🟡" : "🟢"} {alertLevel}
          </div>
        </div>
      </header>

      {/* ── Body ────────────────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── Left sidebar ──────────────────────────────────────────────────── */}
        <aside className="w-60 shrink-0 bg-gray-900 border-r border-gray-800
                          flex flex-col overflow-hidden">

          {/* Left tabs */}
          <div className="flex border-b border-gray-800 shrink-0">
            {LEFT_TABS.map(t => (
              <button key={t.id} onClick={() => setActiveTab(t.id)}
                      className={`flex-1 py-2.5 text-xs font-semibold transition-colors
                        ${activeTab === t.id
                          ? "bg-gray-800 text-white border-b-2 border-blue-500"
                          : "text-gray-500 hover:text-gray-300 hover:bg-gray-800/50"}`}>
                {t.icon} {t.label}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-4">

            {/* ── Settings tab ───────────────────────────────────────────────── */}
            {activeTab === "settings" && (
              <>
                {/* Source selector */}
                <section>
                  <p className="text-xs text-gray-500 font-bold uppercase
                                tracking-wider mb-2.5">
                    📹 Video Source
                  </p>
                  <div className="flex gap-1 mb-2.5 bg-gray-800/60 p-1 rounded-lg">
                    {["camera", "file", "path"].map(t => (
                      <button key={t} onClick={() => setSourceType(t)}
                              className={`flex-1 text-xs py-1 rounded-md font-semibold
                                          capitalize transition-all
                                ${sourceType === t
                                  ? "bg-blue-700 text-white shadow"
                                  : "text-gray-500 hover:text-gray-300"}`}>
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
                                   border-gray-700 rounded-lg px-1 py-1.5 text-white
                                   focus:outline-none focus:border-blue-500"
                      />
                      <button onClick={startCamera}
                              className="flex-1 text-xs py-1.5 rounded-lg bg-blue-700
                                         hover:bg-blue-600 font-semibold transition-colors">
                        ▶ Start Camera
                      </button>
                    </div>
                  )}

                  {sourceType === "file" && (
                    <div className="flex flex-col gap-1.5">
                      <input
                        type="file" ref={fileRef} accept="video/*"
                        className="text-xs text-gray-400
                                   file:mr-2 file:text-xs file:bg-gray-700 file:border-0
                                   file:rounded-lg file:text-gray-200 file:py-1 file:px-2
                                   file:cursor-pointer file:hover:bg-gray-600"
                      />
                      <button onClick={startFile}
                              className="text-xs py-1.5 rounded-lg bg-blue-700
                                         hover:bg-blue-600 font-semibold transition-colors">
                        ▶ Upload & Start
                      </button>
                    </div>
                  )}

                  {sourceType === "path" && (
                    <div className="flex flex-col gap-1.5">
                      <input
                        value={rtspPath}
                        onChange={e => setRtspPath(e.target.value)}
                        placeholder="rtsp://... or /path/to/video"
                        className="text-xs bg-gray-800 border border-gray-700 rounded-lg
                                   px-2.5 py-1.5 text-white placeholder-gray-600
                                   focus:outline-none focus:border-blue-500"
                      />
                      <button onClick={startPath}
                              className="text-xs py-1.5 rounded-lg bg-blue-700
                                         hover:bg-blue-600 font-semibold transition-colors">
                        ▶ Start RTSP
                      </button>
                    </div>
                  )}

                  {isRunning && (
                    <button onClick={stopStream}
                            className="w-full mt-2 text-xs py-1.5 rounded-lg bg-red-900/80
                                       hover:bg-red-800 font-semibold text-red-200
                                       border border-red-800 transition-colors">
                      ⏹ Stop Stream
                    </button>
                  )}
                </section>

                {/* Feature toggles */}
                <section>
                  <p className="text-xs text-gray-500 font-bold uppercase
                                tracking-wider mb-3">
                    Feature Toggles
                  </p>
                  <div className="bg-gray-800/40 rounded-xl p-3 border
                                  border-gray-700/50 flex flex-col gap-4">
                    <Toggle
                      checked={yoloEnabled} onChange={toggleYolo}
                      disabled={toggling.yolo || !isRunning}
                      label="YOLO Detection" color="blue"
                      sublabel="Person tracking + weapons"
                    />
                    <div className="border-t border-gray-700/50" />
                    <Toggle
                      checked={vlmEnabled} onChange={toggleVlm}
                      disabled={toggling.vlm || isSwitching || !isRunning}
                      label="VLM Analysis" color="purple"
                      sublabel="Scene + trigger descriptions"
                    />
                  </div>
                  {!isRunning && (
                    <p className="text-xs text-gray-700 mt-2 text-center">
                      Start a stream first
                    </p>
                  )}
                </section>

                {/* VLM interval */}
                {vlmEnabled && (
                  <section>
                    <p className="text-xs text-gray-500 font-bold uppercase
                                  tracking-wider mb-1.5">
                      🕒 Passive Interval
                    </p>
                    <p className="text-xs text-gray-700 mb-2.5 leading-snug">
                      Passive scene scan. Triggers fire instantly regardless.
                    </p>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setIntervalInput(v => Math.max(2, v - 1))}
                        className="w-8 h-8 rounded-lg bg-gray-700 hover:bg-gray-600
                                   text-white font-bold text-lg flex items-center
                                   justify-center shrink-0 transition-colors">
                        −
                      </button>
                      <input
                        type="number" min={2} max={30}
                        value={intervalInput}
                        onChange={e => setIntervalInput(
                          Math.max(2, Math.min(30, Number(e.target.value)))
                        )}
                        className="flex-1 text-center text-lg font-black bg-gray-800
                                   border border-gray-700 rounded-lg py-1.5 text-white
                                   focus:outline-none focus:border-blue-500"
                      />
                      <button
                        onClick={() => setIntervalInput(v => Math.min(30, v + 1))}
                        className="w-8 h-8 rounded-lg bg-gray-700 hover:bg-gray-600
                                   text-white font-bold text-lg flex items-center
                                   justify-center shrink-0 transition-colors">
                        +
                      </button>
                      <span className="text-xs text-gray-500 shrink-0 w-6">sec</span>
                    </div>
                    <button
                      onClick={saveInterval}
                      disabled={intervalSaving}
                      className="w-full mt-2 text-xs py-1.5 rounded-lg font-semibold
                                 bg-blue-800 hover:bg-blue-700 text-white
                                 disabled:opacity-50 transition-colors">
                      {intervalSaving ? "Saving…" : "✓ Apply"}
                    </button>
                    <p className="text-xs text-gray-700 mt-1 text-center">
                      Min 2s · Max 30s
                    </p>
                  </section>
                )}
              </>
            )}

            {/* ── Triggers tab ───────────────────────────────────────────────── */}
            {activeTab === "triggers" && (
              <div className="flex flex-col gap-3">
                <div className="bg-blue-950/20 border border-blue-900/40 rounded-xl
                                p-3 text-xs text-gray-400 leading-relaxed">
                  <span className="text-yellow-400 font-bold">⚡ Triggers preempt passive scans.</span>
                  {" "}Leave blank to use smart defaults.
                </div>
                {["proximity", "count_change", "weapon"].map(type => (
                  <TriggerPromptEditor
                    key={type}
                    triggerType={type}
                    value={prompts[type]}
                    onChange={v => setPrompts(p => ({ ...p, [type]: v }))}
                    onSave={() => savePrompt(type)}
                    onClear={() => clearPrompt(type)}
                    saving={promptSaving[type]}
                    saved={promptSaved[type]}
                  />
                ))}
              </div>
            )}

          </div>
        </aside>

        {/* ── Center: video feed ─────────────────────────────────────────────── */}
        <main className="flex-1 flex flex-col overflow-hidden min-w-0 bg-black">

          {/* Thin alert banner */}
          {alertLevel !== "CLEAR" && (
            <div className={`flex items-center gap-3 px-4 py-2 shrink-0
                             text-xs font-semibold ${alertBarCfg.bar}`}>
              <span className="shrink-0">
                {alertLevel === "RED" ? "🔴" : "🟡"}
              </span>
              <span className={`truncate ${alertBarCfg.text}`}>
                {status?.reason ?? ""}
              </span>
            </div>
          )}

          {/* Video area */}
          <div className="flex-1 relative flex items-center justify-center overflow-hidden">
            {isRunning ? (
              <>
                <img
                  src={`${API}/video_feed`}
                  className="max-h-full max-w-full object-contain"
                  alt="Live feed"
                />

                {/* Minimal corner badges — no clutter */}
                <div className="absolute bottom-3 left-3 flex gap-1.5 pointer-events-none">
                  {yoloEnabled && (
                    <span className="text-xs px-2.5 py-1 rounded-full font-semibold
                                     bg-blue-900/70 text-blue-200
                                     border border-blue-800/60 backdrop-blur-sm">
                      🎯 YOLO
                    </span>
                  )}
                  {vlmEnabled && (
                    <span className="text-xs px-2.5 py-1 rounded-full font-semibold
                                     bg-purple-900/70 text-purple-200
                                     border border-purple-800/60 backdrop-blur-sm">
                      🧠 VLM
                    </span>
                  )}
                  {!yoloEnabled && !vlmEnabled && (
                    <span className="text-xs px-2.5 py-1 rounded-full font-semibold
                                     bg-gray-800/70 text-gray-500
                                     border border-gray-700/50 backdrop-blur-sm">
                      RAW
                    </span>
                  )}
                </div>

                {/* FPS — bottom right, subtle */}
                {(status?.source_fps ?? 0) > 0 && (
                  <div className="absolute bottom-3 right-3 pointer-events-none">
                    <span className="text-xs text-gray-600 font-mono">
                      {status.source_fps}fps
                    </span>
                  </div>
                )}
              </>
            ) : (
              <div className="flex flex-col items-center gap-4 text-gray-800">
                <span className="text-8xl select-none">📷</span>
                <div className="text-center">
                  <p className="text-base font-bold text-gray-600">No stream active</p>
                  <p className="text-xs text-gray-700 mt-1">
                    Choose a source in ⚙️ Settings and press Start
                  </p>
                </div>
              </div>
            )}
          </div>
        </main>

        {/* ── Right panel ───────────────────────────────────────────────────── */}
        <RightPanel
          status={status}
          alerts={alerts}
          persons={persons}
          vram={vram}
          isRunning={isRunning}
        />
      </div>
    </div>
  )
}