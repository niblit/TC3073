{
  description = "A reproducible evasion and robustness engine for phishing/BEC detectors";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        python = pkgs.python312.override {
          packageOverrides = final: prev: {
            looptime = prev.looptime.overridePythonAttrs (old: {
              doCheck = false;
            });
            wandb = prev.wandb.overridePythonAttrs (old: {
              doCheck = false;
            });
          };
        };
        
        pythonEnv = python.withPackages (ps: with ps; [
          numpy
          pandas
          scikit-learn
          pytest
          joblib
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
          ];
          
          shellHook = ''
            echo "Environment ready for Evasion Engine Research."
            python --version
          '';
        };
      }
    );
}
