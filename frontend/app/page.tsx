"use client";

import { FormEvent, useRef, useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

type TextResult = {
  summary?: string;
  styling_intent?: string;
  keywords?: string[];
  colors?: string[];
  top_pieces?: string[];
  occasions?: string[];
  tone?: string;
};

type ProfileResult = {
  gender?: string;
  skin_tone?: string;
  undertone?: string;
  age_range?: string;
  best_palettes?: string[];
  style_vibes?: string[];
  fit_notes?: string[];
  pieces_to_prioritize?: string[];
  avoid?: string[];
  uplifts?: string[];
};

type AgentProduct = {
  id?: string;
  title?: string;
  brand?: string;
  color?: (string | null)[] | string;
  tags?: string[];
};

type AgentOutfit = {
  title?: string;
  occasion?: string;
  vibe?: string;
};

type AgentClarificationOption = {
  id?: string;
  label?: string;
  short_description?: string;
};

type AgentResponse = {
  stylist_response?: string;
  products?: AgentProduct[];
  outfits?: AgentOutfit[];
  clarification?: {
    question?: string;
    options?: AgentClarificationOption[];
  } | null;
  disambiguation?: {
    options?: unknown[];
  } | null;
  ui_event?: Record<string, unknown>;
};

type AgentUiEvent = {
  type: "clarification_choice";
  payload: string;
};

type Message = {
  id: string;
  role: "user" | "ai";
  content: string;
  data?: TextResult; // For AI messages that have structured data
  agent?: AgentResponse;
};

function TagList({ items }: { items?: string[] }) {
  if (!items || items.length === 0) return <span className="dossier-value">Pending...</span>;
  return (
    <div className="tag-cloud">
      {items.map((item) => (
        <span className="tag" key={item}>
          {item}
        </span>
      ))}
    </div>
  );
}

export default function Home() {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [profileResult, setProfileResult] = useState<ProfileResult | null>(null);
  const [loadingText, setLoadingText] = useState(false);
  const [loadingProfile, setLoadingProfile] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const threadIdRef = useRef<string>(typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `${Date.now()}`);
  const userIdRef = useRef<string>("demo-user");

  const toStrings = (values: (string | null | undefined)[]) =>
    values
      .map((v) => (v ?? "").toString().trim())
      .filter((v): v is string => v.length > 0);

  const mapAgentToTextResult = (agent?: AgentResponse): TextResult | undefined => {
    if (!agent) return undefined;

    const topPieces = toStrings((agent.products ?? []).map((p) => p.title)).slice(0, 8);
    const colorPool = toStrings(
      (agent.products ?? []).flatMap((p) => {
        if (!p.color) return [];
        if (Array.isArray(p.color)) return p.color;
        return [p.color];
      }) as (string | null)[]
    );
    const colors = Array.from(new Set(colorPool)).slice(0, 8);
    const occasions = toStrings((agent.outfits ?? []).map((o) => o.title || o.occasion || o.vibe)).slice(0, 6);
    const keywords = Array.from(new Set(toStrings((agent.products ?? []).flatMap((p) => p.tags ?? [])))).slice(0, 8);

    return {
      summary: agent.stylist_response,
      styling_intent: agent.stylist_response,
      colors,
      top_pieces: topPieces,
      occasions,
      keywords,
      tone: "stylist"
    };
  };

  // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loadingText]);

  const runQuery = async (text: string, uiEvents: AgentUiEvent[] = []) => {
    if (!text.trim() && uiEvents.length === 0) return;

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: text || "..." };
    setMessages((prev) => [...prev, userMsg]);
    setQuery("");
    setLoadingText(true);

    try {
      const response = await fetch(`${apiBase}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: userIdRef.current,
          threadId: threadIdRef.current,
          message: text,
          ui_events: uiEvents
        })
      });

      if (!response.ok) throw new Error("Failed to fetch response");

      const payload: AgentResponse = await response.json();
      const result = mapAgentToTextResult(payload);

      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "ai",
        content: payload.stylist_response ?? result?.styling_intent ?? "Here is what I found for you.",
        data: result,
        agent: payload
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (err) {
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "ai",
        content: "I'm having trouble connecting to the stylist engine. Please try again."
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoadingText(false);
    }
  };

  const handleSearch = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    await runQuery(query);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void runQuery(query);
    }
  };

  const handleUpload = async (file: File) => {
    setLoadingProfile(true);
    try {
      const form = new FormData();
      form.append("image", file);
      const response = await fetch(`${apiBase}/api/analyze/profile`, {
        method: "POST",
        body: form
      });
      if (!response.ok) throw new Error("Profile upload failed");
      const payload = await response.json();
      setProfileResult(payload.data ?? payload);
    } catch (err) {
      console.error(err);
      alert("Could not analyze profile image.");
    } finally {
      setLoadingProfile(false);
    }
  };

  const handleClarificationSelect = async (option: AgentClarificationOption) => {
    if (loadingText) return;
    const label = option.label ?? "Refine my search";
    const payload = option.id ?? option.label ?? "clarification_choice";
    await runQuery(label, [{ type: "clarification_choice", payload }]);
  };

  return (
    <div className="app-container">
      {/* Sidebar - Dossier */}
      <motion.aside
        className="sidebar"
        initial={{ x: -50, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
      >
        <div className="sidebar-header">
          <span>Sortme AI</span>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-label">Your Profile</div>
          {loadingProfile ? (
            <div className="dossier-item">
              <span className="loading-dots" style={{ padding: 0 }}>
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
              </span>
            </div>
          ) : (
            <>
              <div className="dossier-item">
                <span>Gender</span>
                <span className="dossier-value">{profileResult?.gender ?? "-"}</span>
              </div>
              <div className="dossier-item">
                <span>Skin Tone</span>
                <span className="dossier-value">{profileResult?.skin_tone ?? "-"}</span>
              </div>
              <div className="dossier-item">
                <span>Undertone</span>
                <span className="dossier-value">{profileResult?.undertone ?? "-"}</span>
              </div>
            </>
          )}
          <button
            className="upload-btn"
            style={{ marginTop: "1rem" }}
            onClick={() => fileRef.current?.click()}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            Upload Photo
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
          />
        </div>

        {profileResult && (
          <motion.div
            className="sidebar-section"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="sidebar-label">Style DNA</div>
            <div className="dossier-item" style={{ display: 'block' }}>
              <div style={{ marginBottom: '4px', fontSize: '0.8rem', color: 'var(--text-tertiary)' }}>Vibes</div>
              <div className="dossier-value">{profileResult.style_vibes?.join(", ")}</div>
            </div>
            <div className="dossier-item" style={{ display: 'block' }}>
              <div style={{ marginBottom: '4px', fontSize: '0.8rem', color: 'var(--text-tertiary)' }}>Palette</div>
              <div className="dossier-value">{profileResult.best_palettes?.join(", ")}</div>
            </div>
          </motion.div>
        )}
      </motion.aside>

      {/* Main Chat Area */}
      <main className="main-content">
        <div className="chat-scroll-area">
          <div className="chat-container">
            <AnimatePresence mode="popLayout">
              {messages.length === 0 ? (
                <motion.div
                  key="hero"
                  className="hero-empty"
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.4 }}
                >
                  <h1 className="hero-title">What are we styling today?</h1>
                  <p className="hero-subtitle">
                    Ask for outfit ideas, color matches, or upload a photo to define your personal style profile.
                  </p>
                </motion.div>
              ) : (
                messages.map((msg) => (
                  <motion.div
                    key={msg.id}
                    className={`message-row ${msg.role}`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, ease: "easeOut" }}
                  >
                    <div className={`avatar ${msg.role}`}>
                      {msg.role === "ai" ? "Sort" : "You"}
                    </div>
                    <div className="message-content">
                      {msg.content}
                      {msg.agent?.clarification?.question && (msg.agent?.clarification?.options?.length ?? 0) > 0 && (
                        <motion.div
                          className="clarification-card"
                          initial={{ opacity: 0, scale: 0.95 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: 0.2 }}
                        >
                          <div className="clarification-question">{msg.agent.clarification.question}</div>
                          <div className="clarification-options">
                            {msg.agent.clarification.options?.map((option, idx) => (
                              <button
                                type="button"
                                key={option.id ?? option.label ?? `clar-option-${idx}`}
                                className="clarification-chip"
                                onClick={() => handleClarificationSelect(option)}
                                disabled={loadingText}
                              >
                                <span className="clarification-chip-label">{option.label ?? "Option"}</span>
                                {option.short_description && (
                                  <span className="clarification-chip-sub">{option.short_description}</span>
                                )}
                              </button>
                            ))}
                          </div>
                        </motion.div>
                      )}
                      {msg.data && (
                        <motion.div
                          className="result-grid"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.3 }}
                        >

                        </motion.div>
                      )}
                    </div>
                  </motion.div>
                ))
              )}
            </AnimatePresence>

            {loadingText && (
              <motion.div
                className="message-row ai"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <div className="avatar ai">Sort</div>
                <div className="message-content">
                  <div className="loading-dots">
                    <div className="dot"></div>
                    <div className="dot"></div>
                    <div className="dot"></div>
                  </div>
                </div>
              </motion.div>
            )}
            <div ref={chatEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="input-area-wrapper">
          <motion.div
            className="input-container"
            initial={{ y: 50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.5, type: "spring", stiffness: 100 }}
          >
            <form onSubmit={handleSearch}>
              <textarea
                className="chat-input"
                placeholder="Describe an occasion, vibe, or piece of clothing..."
                rows={1}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
              />
              <div className="input-actions">
                <button
                  type="button"
                  className="icon-button"
                  onClick={() => fileRef.current?.click()}
                  title="Upload Profile Image"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                    <circle cx="8.5" cy="8.5" r="1.5" />
                    <polyline points="21 15 16 10 5 21" />
                  </svg>
                </button>
                <button
                  type="submit"
                  className={`icon-button ${query.trim() ? 'send-btn' : ''}`}
                  disabled={!query.trim() || loadingText}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13"></line>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                  </svg>
                </button>
              </div>
            </form>
          </motion.div>
        </div>
      </main>
    </div>
  );
}
