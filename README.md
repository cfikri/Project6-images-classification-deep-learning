You are a volunteer for the animal protection association in your neighborhood, ***Le Refuge***. In fact, that's how you found your perfect companion, Snooky. Now, you're wondering how you can give back and help the association.

While discussing with another volunteer, you learn that their database of shelter animals is growing, and they don’t always have time to catalog the images of the animals they’ve accumulated over the years.
They would like to have an algorithm capable of classifying images based on the breed of the dog in the picture.

Since the volunteers haven’t yet gathered all the images scattered across their hard drives, you decide to train your algorithm using the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).

**The association asks you to develop a ***dog breed detection algorithm*** to speed up their indexing process.**
Since you have little experience in this field, you reach out to your friend Luc, an expert in image classification. He replies via email:

"Hey,

I've put together some tips to help you get started with this project!

First, I recommend **preprocessing the images** using specific techniques (e.g., whitening, equalization, and possibly resizing) and applying **data augmentation** (mirroring, cropping, etc.).

Next, I suggest implementing two state-of-the-art approaches using CNNs and comparing them in terms of processing time and accuracy:

The first approach involves **building your own CNN** by drawing inspiration from existing CNN architectures. After training your initial model, you’ll refine it by optimizing hyperparameters (related to the model’s layers, compilation, and execution). Don't forget to use data augmentation to compensate for the limited number of dog images per class and improve performance.

The second approach relies on **transfer learning**, using a pre-trained network and adapting it to your problem.

Concerning transfer learning, The first essential step is to retrain the final layers to predict only the classes you need.
You can also modify the network’s structure (e.g. remove certain layers) or fine-tune the entire model with a very low learning rate to gradually adjust the weights to your problem. This approach takes longer but can optimize performance.

Notice that training a CNN (even partially) requires a lot of computing power. If the association's older computer processor isn’t enough, here are some solutions:

Limit the dataset by selecting only three dog breeds. This will allow you to test your approach and model design before scaling up, if resources permit.
Use the computer’s GPU (though installation is a bit complex, and the machine will be unusable during training).
If no GPU is available, work in the cloud with [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb), which offers free or low-cost GPU-powered CNN training.
Finally, I suggest creating a **demo of your dog breed prediction engine**, either locally on your PC via a notebook or using Streamlit.

Good luck!

Luc"

**Main skills involved:**

- Image preprocessing techniques
- CNN hyperparameter optimization
- Transfer learning
- Creating and deploying an application with Streamlit and Render
- Writing unit tests
- Continuous integration and deployment
