FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# Set the correct working directory
WORKDIR /workspaces/xbandits

# Copy requirements.txt and setup.py into the container
COPY requirements.txt setup.py ./

# Create a Python 3.11 environment
RUN conda create -n xbandits python=3.11

# Initialize conda for bash and activate the environment
RUN conda init bash && \
    echo "conda activate xbandits" >> ~/.bashrc

# Install pip requirements and the xbandits package
SHELL ["/bin/bash", "--login", "-c"]
RUN conda activate xbandits && \
    pip install -r requirements.txt && \
    pip install -e .

# Set the default command to bash
CMD ["/bin/bash"]