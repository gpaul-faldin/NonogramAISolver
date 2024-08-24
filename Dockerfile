# Use an official TensorFlow image as the base image
FROM tensorflow/tensorflow:latest-gpu

# Install Jupyter and other required packages
RUN pip install jupyter matplotlib numpy scikit-learn seaborn matplotlib pandas 

# Set the working directory in the container
WORKDIR /app

# Expose the port Jupyter will run on
EXPOSE 8888

# Command to start Jupyter when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
