import {
  Zap,
  BrainCircuit,
  Languages,
  Volume2,
} from "lucide-react";

function Features() {
  const features = [
    {
      icon: <Zap size={36} />,
      title: "Real-Time Recognition",
      description:
        "Recognize Indian Sign Language instantly using MediaPipe and Machine Learning.",
    },
    {
      icon: <BrainCircuit size={36} />,
      title: "AI Powered",
      description:
        "Built with a modular architecture ready for LSTM and Transformer models.",
    },
    {
      icon: <Languages size={36} />,
      title: "Multilingual",
      description:
        "Translate recognized signs into Kannada and future multilingual outputs.",
    },
    {
      icon: <Volume2 size={36} />,
      title: "Speech Ready",
      description:
        "Future versions will generate natural speech from translated text.",
    },
  ];

  return (
    <section className="bg-zinc-950 py-28">

      <div className="mx-auto max-w-7xl px-8">

        <h2 className="text-center font-['Poppins'] text-5xl font-bold">

          Why Choose ISL Translate?

        </h2>

        <p className="mx-auto mt-6 max-w-2xl text-center text-zinc-400">

          Making communication more accessible through
          Artificial Intelligence and Computer Vision.

        </p>

        <div className="mt-20 grid gap-8 md:grid-cols-2 lg:grid-cols-4">

          {features.map((feature) => (

            <div
              key={feature.title}
              className="rounded-3xl border border-zinc-800 bg-zinc-900/60 p-8 transition duration-300 hover:-translate-y-2 hover:border-violet-500 hover:shadow-[0_0_40px_rgba(124,58,237,0.25)]"
            >

              <div className="mb-6 text-violet-400">

                {feature.icon}

              </div>

              <h3 className="mb-4 font-['Poppins'] text-2xl font-semibold">

                {feature.title}

              </h3>

              <p className="leading-8 text-zinc-400">

                {feature.description}

              </p>

            </div>

          ))}

        </div>

      </div>

    </section>
  );
}

export default Features;