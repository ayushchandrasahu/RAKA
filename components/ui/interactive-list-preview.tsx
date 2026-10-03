// Built using Hyperiux Vault: https://vault.hyperiux.com

"use client";

import { useEffect, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import gsap from "gsap";

export interface InteractiveListItem {
  client: string;
  platform?: string;
  services: string;
  img: string;
}

export interface InteractiveListPreviewProps {
  items?: InteractiveListItem[];
  /** Scale multiplier for the hover preview image. */
  imageSize?: number;
  /** Preview image reveal / hide duration (seconds). */
  duration?: number;
  /** Highlight bar + row text transition smoothing (seconds). */
  smoothness?: number;
  /** Pointer-follow smoothing; higher tracks faster. */
  lerp?: number;
  /** Background color of the list surface. */
  bgColor?: string;
  className?: string;
}

const DEFAULT_IMAGE_Z_INDEX = 10;
const DEFAULT_IMAGE_SIZE = 1;
const DEFAULT_DURATION = 0.6;
const DEFAULT_SMOOTHNESS = 0.35;
const DEFAULT_LERP = 0.18;

const DEFAULT_ITEMS: InteractiveListItem[] = [
  {
    client: "AURORA UI",
    platform: "NEXT.JS",
    services: "Motion Design, GSAP, Page Transitions, UI Systems",
    img: "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=1000&auto=format&fit=crop"
  },
  {
    client: "NEON FLOW",
    platform: "REACT",
    services: "Interactive UI, Scroll Animations, Effects Library",
    img: "https://images.unsplash.com/photo-1550745165-9bc0b252726f?q=80&w=1000&auto=format&fit=crop"
  },
  {
    client: "GLASSMORPH",
    platform: "NEXT.JS",
    services: "Glass UI, Components, Motion Architecture",
    img: "https://images.unsplash.com/photo-1634017839464-5c339ebe3cb4?q=80&w=1000&auto=format&fit=crop"
  },
  {
    client: "VOID SYSTEM",
    platform: "CUSTOM WEBGL",
    services: "Shaders, Creative Development, Visual Effects",
    img: "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=1000&auto=format&fit=crop"
  }
];

export function InteractiveListPreview({
  items = DEFAULT_ITEMS,
  imageSize = DEFAULT_IMAGE_SIZE,
  duration = DEFAULT_DURATION,
  smoothness = DEFAULT_SMOOTHNESS,
  lerp = DEFAULT_LERP,
  bgColor = "transparent",
  className = ""
}: InteractiveListPreviewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const previewRef = useRef<HTMLDivElement>(null);
  const [activeItem, setActiveItem] = useState<InteractiveListItem | null>(null);
  const [isHovered, setIsHovered] = useState(false);

  const mousePos = useRef({ x: 0, y: 0 });
  const previewPos = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      mousePos.current = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      };
    };

    const container = containerRef.current;
    if (container) {
      container.addEventListener("mousemove", handleMouseMove);
    }

    let animationFrameId: number;
    const updatePreview = () => {
      if (previewRef.current) {
        previewPos.current.x += (mousePos.current.x - previewPos.current.x) * lerp;
        previewPos.current.y += (mousePos.current.y - previewPos.current.y) * lerp;

        gsap.set(previewRef.current, {
          x: previewPos.current.x,
          y: previewPos.current.y,
          xPercent: -50,
          yPercent: -50
        });
      }
      animationFrameId = requestAnimationFrame(updatePreview);
    };

    updatePreview();

    return () => {
      if (container) {
        container.removeEventListener("mousemove", handleMouseMove);
      }
      cancelAnimationFrame(animationFrameId);
    };
  }, [lerp]);

  useEffect(() => {
    if (!previewRef.current) return;

    if (isHovered && activeItem) {
      gsap.to(previewRef.current, {
        opacity: 1,
        scale: imageSize,
        duration: duration,
        ease: "power3.out"
      });
    } else {
      gsap.to(previewRef.current, {
        opacity: 0,
        scale: imageSize * 0.8,
        duration: duration * 0.7,
        ease: "power3.in"
      });
    }
  }, [isHovered, activeItem, imageSize, duration]);

  const handleMouseEnterRow = (item: InteractiveListItem) => {
    setActiveItem(item);
    setIsHovered(true);
  };

  const handleMouseLeaveList = () => {
    setIsHovered(false);
  };

  return (
    <div
      ref={containerRef}
      onMouseLeave={handleMouseLeaveList}
      className={`relative w-full overflow-hidden select-none ${className}`}
      style={{ backgroundColor: bgColor }}
    >
      {/* Floating Preview Image */}
      <div
        ref={previewRef}
        className="pointer-events-none absolute left-0 top-0 hidden md:block rounded-xl overflow-hidden shadow-2xl border border-white/20 transition-opacity"
        style={{
          width: `${300 * imageSize}px`,
          height: `${200 * imageSize}px`,
          zIndex: DEFAULT_IMAGE_Z_INDEX,
          opacity: 0,
          transform: "translate(-50%, -50%)"
        }}
      >
        {activeItem && (
          <img
            src={activeItem.img}
            alt={activeItem.client}
            className="w-full h-full object-cover"
          />
        )}
      </div>

      {/* List items */}
      <div className="flex flex-col divide-y divide-white/10">
        {items.map((item, index) => {
          const isCurrent = activeItem?.client === item.client && isHovered;
          return (
            <div
              key={index}
              onMouseEnter={() => handleMouseEnterRow(item)}
              className="group relative flex flex-col md:flex-row md:items-center justify-between py-6 px-4 md:px-8 cursor-pointer transition-colors"
              style={{
                transitionDuration: `${smoothness}s`
              }}
            >
              {/* Highlight bar */}
              <div
                className={`absolute inset-0 bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none`}
                style={{ transitionDuration: `${smoothness}s` }}
              />

              <div className="flex items-center gap-4 z-0">
                <span className="text-xs font-mono text-zinc-500 group-hover:text-zinc-300">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <h3
                  className={`text-xl md:text-2xl font-bold tracking-tight transition-colors ${
                    isCurrent ? "text-white translate-x-2" : "text-zinc-300 group-hover:text-white"
                  }`}
                  style={{ transitionDuration: `${smoothness}s` }}
                >
                  {item.client}
                </h3>
                {item.platform && (
                  <span className="text-xs px-2 py-0.5 rounded border border-white/20 text-zinc-400 font-mono">
                    {item.platform}
                  </span>
                )}
              </div>

              <div className="mt-2 md:mt-0 text-sm text-zinc-400 group-hover:text-zinc-200 z-0">
                {item.services}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default InteractiveListPreview;
