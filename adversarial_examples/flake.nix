{
  description = "A reproducible evasion and robustness engine for phishing/BEC detectors";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python312
            pkgs.uv
            # Required C libraries for pre-compiled PyPI wheels
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];
          
          shellHook = ''
            # Tell the dynamic linker where to find the C/C++ libraries
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ]}:$LD_LIBRARY_PATH"

            export NLTK_DATA=$PWD/.nltk_data
            mkdir -p $NLTK_DATA
            
            # Create a virtual environment if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "Setting up fast virtual environment via uv..."
              uv venv
              source .venv/bin/activate
              
              # Install deps lightning fast
              uv pip install pytest nltk spacy sentence-transformers
              
              # Let spacy fetch the correct model version dynamically
              python -m spacy download en_core_web_sm
            else
              source .venv/bin/activate
            fi

            # NLTK setup
            python -c "import nltk; nltk.download('wordnet', download_dir='$NLTK_DATA', quiet=True); nltk.download('averaged_perceptron_tagger_eng', download_dir='$NLTK_DATA', quiet=True); nltk.download('stopwords', download_dir='$NLTK_DATA', quiet=True); nltk.download('punkt_tab', download_dir='$NLTK_DATA', quiet=True)"
            python --version
          '';
        };
      }
    );
}