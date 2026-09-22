;;; guix.scm --- Reproducible development shell for the qlora-wgpu project.
;;;
;;; A Rust + WGPU library implementing QLoRA operations (NF4 quantization,
;;; LoRA adapters, fused QLoRA linear), callable from Rust and Python.
;;;
;;; Usage:
;;;   guix shell -f guix.scm              # enter the dev shell
;;;   guix shell -f guix.scm -- cargo test --offline -p qlora-core
;;;   guix shell -f guix.scm -- cargo build -p qlora-wgpu   # needs network
;;;
;;; Why the tools live in `propagated-inputs': plain `guix shell -f
;;; guix.scm' (without -D) installs the package plus its propagated inputs,
;;; but NOT its plain `inputs' (those need -D). Propagating everything is
;;; what makes the exact command `guix shell -f guix.scm' work.
;;;
;;; Inside the shell you build with Cargo / maturin as usual:
;;;   cd qlora-core   && cargo test
;;;   cd ../qlora-wgpu && cargo build            # downloads wgpu from crates.io
;;;   cd ../qlora-python && maturin develop      # needs a venv, see README.md
;;;
;;; NOTE (TLS): Cargo needs a CA bundle to fetch from crates.io.  If fetch
;;; fails with a certificate error inside the shell, export:
;;;   export SSL_CERT_FILE="$GUIX_ENVIRONMENT/etc/ssl/certs/ca-certificates.crt"
;;; (`nss-certs' below provides that bundle.)

(use-modules (guix packages)
             (guix gexp)
             ((guix licenses) #:prefix license:)
             (guix build-system trivial)
             (gnu packages base)
             (gnu packages build-tools)
             (gnu packages certs)
             (gnu packages check)
             (gnu packages commencement)
             (gnu packages gl)
             (gnu packages machine-learning)
             (gnu packages nss)
             (gnu packages pkg-config)
             (gnu packages python)
             (gnu packages python-xyz)
             (gnu packages rust)
             (gnu packages tls)
             (gnu packages version-control)
             (gnu packages vulkan))

(package
  (name "qlora-wgpu-dev-shell")
  (version "0.1.0")
  (source #f)
  (build-system trivial-build-system)
  (arguments
   (list #:modules '((guix build utils))
         #:builder
         #~(begin
             (use-modules (guix build utils))
             (mkdir-p #$output)
             (call-with-output-file (string-append #$output "/README")
               (lambda (port)
                 (display "qlora-wgpu reproducible dev shell (see guix.scm).\n"
                          port)))
             #t)))
  ;; All tools are propagated so that plain `guix shell -f guix.scm' works.
  (propagated-inputs
   (list
    ;; Rust toolchain (rustc, cargo) + C linker + build helpers.
    rust
    gcc-toolchain
    pkg-config
    git
    ;; Python side: interpreter, NumPy/pytest for tests, maturin for the
    ;; PyO3 extension module in qlora-python/.
    python
    python-numpy
    python-pytest
    ;; PyTorch: only for the .pt boundary tools (tools/pt2npz.py,
    ;; tools/npz2pt.py). The Rust side never touches pickle.
    python-pytorch
    ;; Model parity reference + safetensors fixtures (SmolLM experiments).
    python-transformers
    python-safetensors
    maturin
    ;; GPU side: Vulkan loader + headers (WGPU/Vulkan backend) and Mesa
    ;; (includes lavapipe, so WGPU also runs headless on the CPU).
    vulkan-loader
    vulkan-headers
    mesa
    ;; TLS for Cargo / pip fetches.
    openssl
    nss-certs))
  (synopsis "Development shell for the qlora-wgpu QLoRA library")
  (description
   "This package exists to provide a reproducible development environment
for qlora-wgpu, a Rust + WGPU library implementing QLoRA operations (NF4
block-wise quantization with double quantization, LoRA adapters, and fused
QLoRA linear layers) with Rust and Python APIs.  Entering @code{guix shell
-f guix.scm} makes the Rust toolchain, Python, maturin, and the Vulkan/Mesa
GPU stack available so the project can be built with Cargo and maturin.")
  (home-page "https://example.org/qlora-wgpu")
  (license (list license:expat license:asl2.0)))
