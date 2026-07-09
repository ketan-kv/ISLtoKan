function Navbar() {
  return (
    <header className="fixed top-0 left-0 z-50 w-full border-b border-zinc-800/50 bg-zinc-950/70 backdrop-blur-md">

      <nav className="mx-auto flex h-20 max-w-7xl items-center justify-between px-8">

        {/* Logo */}

        <div className="flex items-center gap-3">

          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-violet-600 text-xl">

            🤟

          </div>

          <div>

            <h1 className="font-['Poppins'] text-xl font-bold">

              SignBridge

            </h1>

            <p className="text-xs text-zinc-400">

              AI Translation Platform

            </p>

          </div>

        </div>

        {/* Navigation */}

        <div className="hidden items-center gap-8 text-sm text-zinc-400 md:flex">

          <a href="#" className="transition hover:text-white">

            Home

          </a>

          <a href="#" className="transition hover:text-white">

            Technology

          </a>

          <a href="#" className="transition hover:text-white">

            Roadmap

          </a>

          <a href="#" className="transition hover:text-white">

            About

          </a>

        </div>

        {/* Button */}

        <button className="rounded-xl bg-violet-600 px-6 py-3 font-medium transition duration-300 hover:scale-105 hover:bg-violet-500">

          Launch App

        </button>

      </nav>

    </header>
  );
}

export default Navbar;