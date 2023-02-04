FROM mambaorg/micromamba
COPY . .
RUN micromamba install -f environment.yaml -n base --yes


