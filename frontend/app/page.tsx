"use client";

import { HeroScene } from "@/components/HeroScene";
import clsx from "clsx";
import { motion } from "framer-motion";
import Image from "next/image";
import { useMemo, useRef, useState } from "react";

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

type GalleryCard = { title: string; src: string; badge: string };

const heroGallery: GalleryCard[] = [
  {
    title: "Editorial off-duty",
    badge: "Muted pastels",
    src: "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?auto=format&fit=crop&w=800&q=80"
  },
  {
    title: "City layers",
    badge: "Metallic lift",
    src: "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?auto=format&fit=crop&w=800&q=80"
  },
  {
    title: "Weekend ease",
    badge: "Soft neutrals",
    src: "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&w=800&q=80"
  }
];

const vibePills = ["Minimal luxe", "Soft street", "Gallery opening", "Muted tailoring", "Beach to bar"];

function PillRail({ items }: { items: string[] }) {
  return (
    <div className="meta-row">
      {items.map((item) => (
        <span className="pill" key={item}>
          {item}
        </span>
      ))}
    </div>
  );
}

function TagList({ items }: { items?: string[] }) {
  if (!items || items.length === 0) return <span className="muted">Awaiting insights…</span>;
  return (
    <div className="list-row">
      {items.map((item) => (
        <span className="tag" key={item}>
          {item}
        </span>
      ))}
    </div>
  );
}

function GalleryRail() {
  return (
    <div className="status-row" aria-label="Inspiration rail">
      {heroGallery.map((card) => (
        <div className="status-chip" key={card.title}>
          <Image src={card.src} alt={card.title} width={42} height={42} style={{ borderRadius: 12 }} />
          <div>
            <div className="stat">{card.title}</div>
            <div className="helper">{card.badge}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function Home() {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
  const [query, setQuery] = useState("Studio-ready looks for an autumn city weekend");
  const [textResult, setTextResult] = useState<TextResult | null>(null);
  const [profileResult, setProfileResult] = useState<ProfileResult | null>(null);
  const [uploadLabel, setUploadLabel] = useState<string>("Upload profile image");
  const [loadingText, setLoadingText] = useState(false);
  const [loadingProfile, setLoadingProfile] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const resolvedSummary = textResult?.summary ?? textResult?.styling_intent;

  const headingAccent = useMemo(
    () =>
      ["persona", "driven", "fashion"].map((word, idx) => (
        <span
          key={word}
          style={{
            background: idx % 2 === 0 ? "linear-gradient(120deg,#5d7df5,#ca337c)" : "linear-gradient(120deg,#1d274c,#5d7df5)",
            WebkitBackgroundClip: "text",
            color: "transparent"
          }}
        >
          {word}
        </span>
      )),
    []
  );

  const handleSearch = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    setLoadingText(true);
    try {
      const response = await fetch(`${apiBase}/api/analyze/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
      });
      if (!response.ok) {
        throw new Error("Search failed. Please check the backend service.");
      }
      const payload = await response.json();
      setTextResult(payload.data ?? payload);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoadingText(false);
    }
  };

  const handleFileInput = () => {
    fileRef.current?.click();
  };

  const handleUpload = async (file: File) => {
    setError(null);
    setLoadingProfile(true);
    setUploadLabel(file.name);
    try {
      const form = new FormData();
      form.append("image", file);
      const response = await fetch(`${apiBase}/api/analyze/profile`, {
        method: "POST",
        body: form
      });
      if (!response.ok) {
        throw new Error("Profile analysis failed. Please verify backend is running.");
      }
      const payload = await response.json();
      setProfileResult(payload.data ?? payload);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoadingProfile(false);
    }
  };

  const onFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    void handleUpload(file);
  };

  return (
    <div className="page-shell">
      <header className={clsx("nav-bar", "glass")}>
        <div className="nav-left">
          <div className="orb" />
          <span className="brand">Sortme AI</span>
          <div className="pill pill-ghost">Fashion discovery OS</div>
        </div>
        <div className="nav-links">
          {["Looks", "Fits", "Palette", "Wardrobe"].map((item) => (
            <span className="pill" key={item}>
              {item}
            </span>
          ))}
        </div>
        <div className="nav-actions">
          <button className="button chip">Sign in</button>
          <button className="button primary">Launch Studio</button>
        </div>
      </header>

      <div className="hero-grid">
        <motion.section
          className={clsx("hero-copy", "glass")}
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <span className="gradient-chip">
            <span>GPT-5-mini powered</span> • Vision + taste
          </span>
          <h1>
            Persona-{headingAccent[0]} {headingAccent[1]} {headingAccent[2]}
          </h1>
          <p>Upload a profile, ask in natural language, and watch Sortme AI surface tailored looks, palettes, and must-have pieces.</p>
          <PillRail items={vibePills} />

          <div className={clsx("search-panel", "glass")}>
            <div className="action-grid">
              <form className="search-form" onSubmit={handleSearch}>
                <input
                  className="text-input"
                  placeholder="Search looks, moods, or exact pieces"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <button className="primary-btn" type="submit" disabled={loadingText}>
                  {loadingText ? "Asking..." : "Ask Sortme AI"}
                </button>
              </form>

              <div
                className="upload-tile"
                role="button"
                onClick={handleFileInput}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && handleFileInput()}
                tabIndex={0}
              >
                <div className="upload-icon">+</div>
                <div>
                  <div className="stat">{loadingProfile ? "Analyzing profile…" : "Profile-ready in one tap"}</div>
                  <div className="helper">{uploadLabel}</div>
                </div>
                <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={onFileChange} />
              </div>
            </div>
            <GalleryRail />
            {error && <div className="error">{error}</div>}
          </div>
        </motion.section>

        <motion.section
          className="visual-panel"
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <div className="canvas-card">
            <HeroScene />
          </div>
          <div className="floating-panel">
            <h4>Profile dossier</h4>
            <div className="list-row">
              <span className="pill">{profileResult?.gender ?? "Gender: pending"}</span>
              <span className="pill">{profileResult?.skin_tone ?? "Skin tone: pending"}</span>
              <span className="pill">{profileResult?.undertone ?? "Undertone: pending"}</span>
            </div>
            <div className="meta-row">
              {(profileResult?.style_vibes ?? ["Soft street", "Editorial ease", "Gallery ready"]).map((vibe) => (
                <span key={vibe} className="tag">
                  {vibe}
                </span>
              ))}
            </div>
            <div className="helper">Upload a clear portrait; we infer palette, vibe, and fit guidance automatically.</div>
          </div>
        </motion.section>
      </div>

      <section className="insight-grid">
        <div className={clsx("panel", "glass")}>
          <h3>Search intelligence</h3>
          <p className="helper">Structured results from the OpenAI Responses API (text-to-styling).</p>
          <div className="mini-card">
            <div className="stat">{resolvedSummary ?? "No search yet"}</div>
            <div className="helper">{textResult?.tone ?? textResult?.styling_intent ?? "Try a prompt like “Brunch-ready monochrome in linen”"}</div>
          </div>

          <div className="result-grid">
            <div className="mini-card">
              <h5>Keywords</h5>
              <TagList items={textResult?.keywords} />
            </div>
            <div className="mini-card">
              <h5>Colors</h5>
              <TagList items={textResult?.colors} />
            </div>
            <div className="mini-card">
              <h5>Pieces</h5>
              <TagList items={textResult?.top_pieces} />
            </div>
            <div className="mini-card">
              <h5>Occasions</h5>
              <TagList items={textResult?.occasions} />
            </div>
          </div>
        </div>

        <div className={clsx("panel", "glass")}>
          <h3>Profile insights</h3>
          <p className="helper">Vision-powered GPT-5-mini analysis from your profile upload.</p>
          <div className="result-grid">
            <div className="mini-card">
              <h5>Palette</h5>
              <TagList items={profileResult?.best_palettes} />
            </div>
            <div className="mini-card">
              <h5>Vibes</h5>
              <TagList items={profileResult?.style_vibes} />
            </div>
            <div className="mini-card">
              <h5>Fit notes</h5>
              <TagList items={profileResult?.fit_notes} />
            </div>
            <div className="mini-card">
              <h5>Priorities</h5>
              <TagList items={profileResult?.pieces_to_prioritize ?? profileResult?.uplifts} />
            </div>
            <div className="mini-card">
              <h5>Avoid</h5>
              <TagList items={profileResult?.avoid} />
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
