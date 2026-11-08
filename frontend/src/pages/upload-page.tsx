import Background from "@/components/background";
import Navbar from "@/layouts/navbar";
import Footer from "@/layouts/footer";
import { UploadForm } from "@/features/upload";

/** Workspace: upload a session to build its dashboards. */
export default function UploadPage() {
  return (
    <div className="relative flex min-h-screen flex-col text-ink">
      <Background />

      <div className="relative z-10 flex min-h-screen flex-col">
        <Navbar />

        <main className="flex-grow px-6 pt-32 pb-16">
          <div className="mx-auto w-full max-w-2xl">
            <h1 className="mb-2 text-center font-display text-5xl tracking-tight text-ink">Workspace</h1>
            <p className="mb-10 text-center text-ink-soft">Upload a pitching session to build its dashboards.</p>
            <UploadForm />
          </div>
        </main>

        <Footer />
      </div>
    </div>
  );
}
