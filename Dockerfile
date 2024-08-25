# Use an official Python runtime as a parent image
FROM python:3.12
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app


# Install packages
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install Jupyter Notebook
RUN pip install jupyter

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

 
