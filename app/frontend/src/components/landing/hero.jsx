function Hero() {
  return (
    <section className="relative flex min-h-screen items-center justify-center overflow-hidden px-6">

      {/* Background Glow */}

      <div className="absolute h-[700px] w-[700px] rounded-full bg-violet-600/20 blur-[180px]" />

      <div className="relative z-10 mx-auto max-w-5xl text-center">

        <span className="rounded-full border border-violet-500/40 bg-violet-500/10 px-5 py-2 text-sm text-violet-300">

          Real-Time AI Powered Sign Language Translation

        </span>

        <h1 className="mt-8 font-['Poppins'] text-6xl font-bold leading-tight md:text-8xl">

          Breaking

          <br />

          <span className="text-violet-400">

            Communication

          </span>

          <br />

          Barriers

        </h1>

        <p className="mx-auto mt-8 max-w-3xl text-xl leading-9 text-zinc-400">

          Translate Indian Sign Language into Kannada using
          Computer Vision, Machine Learning, and Artificial
          Intelligence in real time.

        </p>

        <div className="mt-14 flex flex-wrap justify-center gap-5">

          <button className="rounded-xl bg-violet-600 px-8 py-4 font-semibold transition hover:scale-105 hover:bg-violet-500">

            Start Translation

          </button>

          <button className="rounded-xl border border-zinc-700 px-8 py-4 transition hover:border-violet-500 hover:bg-zinc-900">

            Learn More

          </button>

        </div>

      </div>

    </section>
  );
}

export default Hero;