import { InteractiveListPreview } from "../components/ui/interactive-list-preview";

export default function App() {
  return (
    <div className="min-h-screen bg-zinc-950 text-white flex flex-col justify-center items-center p-6 md:p-12">
      <div className="w-full max-w-5xl space-y-8">
        <div className="space-y-2 border-b border-zinc-800 pb-6">
          <span className="text-xs uppercase tracking-widest text-violet-400 font-mono">
            Hyperiux Vault Component
          </span>
          <h1 className="text-3xl md:text-5xl font-extrabold tracking-tight">
            Interactive List Preview
          </h1>
          <p className="text-zinc-400 text-sm md:text-base">
            Hover over the list rows to view the smooth GSAP-animated floating preview.
          </p>
        </div>

        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 backdrop-blur overflow-hidden p-2 md:p-6 shadow-2xl">
          <InteractiveListPreview />
        </div>
      </div>
    </div>
  );
}
