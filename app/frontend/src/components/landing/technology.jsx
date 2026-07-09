import {
  Camera,
  Hand,
  BrainCircuit,
  Languages,
  Volume2,
  ArrowDown,
} from "lucide-react";

function Technology() {
  const pipeline = [
    {
      icon: <Camera size={34} />,
      title: "Camera",
      subtitle: "Live Webcam Feed",
    },
    {
      icon: <Hand size={34} />,
      title: "MediaPipe",
      subtitle: "Hand Landmark Extraction",
    },
    {
      icon: <BrainCircuit size={34} />,
      title: "Recognition",
      subtitle: "Random Forest → LSTM → Transformer",
    },
    {
      icon: <Languages size={34} />,
      title: "Translation",
      subtitle: "Kannada + Future Languages",
    },
    {
      icon: <Volume2 size={34} />,
      title: "Speech",
      subtitle: "Future Text-to-Speech",
    },
  ];

  return (
    <section className="bg-zinc-950 py-32">

      <div className="mx-auto max-w-6xl px-8">

        <h2 className="text-center font-['Poppins'] text-5xl font-bold">

          How It Works

        </h2>

        <p className="mx-auto mt-5 max-w-3xl text-center text-zinc-400">

          A modular AI pipeline designed for scalability,
          multilingual translation, and future Transformer models.

        </p>

        <div className="mt-24 flex flex-col items-center">

          {pipeline.map((step, index) => (
            <div
              key={step.title}
              className="flex flex-col items-center"
            >

              <div className="w-72 rounded-3xl border border-zinc-800 bg-zinc-900/60 p-8 text-center transition duration-300 hover:border-violet-500 hover:shadow-[0_0_40px_rgba(124,58,237,.2)]">

                <div className="mb-5 flex justify-center text-violet-400">

                  {step.icon}

                </div>

                <h3 className="font-['Poppins'] text-2xl font-semibold">

                  {step.title}

                </h3>

                <p className="mt-2 text-zinc-400">

                  {step.subtitle}

                </p>

              </div>

              {index < pipeline.length - 1 && (
                <ArrowDown
                  size={34}
                  className="my-6 text-violet-500"
                />
              )}

            </div>
          ))}

        </div>

      </div>

    </section>
  );
}

export default Technology;