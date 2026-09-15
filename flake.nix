{
  description = "Comfortable Python computer-vision and 3D development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python312;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python
            uv
            ruff
          ];

          env = {
            # Make Python tooling and interactive sessions use the shell's
            # interpreter and keep bytecode/cache files out of the repo.
            PYTHONUNBUFFERED = "1";
            PYTHONDONTWRITEBYTECODE = "1";
            UV_PYTHON = "${python}/bin/python";
            UV_PROJECT_ENVIRONMENT = "$PWD/.venv";
            UV_CACHE_DIR = ".uv-cache";
          };
        };
      }
    );
}
