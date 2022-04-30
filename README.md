# paintmath-backend
Back end and model used in PaintMath. [open app↗️](https://paintmath.herokuapp.com/)
## Back end
Back end was developed in python using the Flask framework. It's quite simple and only contains one endpoint:
>### `POST` /predict
>Classifies image into an operand.
>#### _Parameters_
>Image bytes encoded with base64
>*Example:* `iVBORw0KGgoAAAANSUhEUgAAAX8AAAD3CAYAAAD10FRmAAAAOXRFWHRTb2Z...`
>#### _Returns_
>A JSON object with the predicted operand (0, 1, ..., +, -, ...)
>*Example:* `{'class_name': '7'}`

## Model
Model was developed using PyTorch framework. It is made up of two convolutional and one fully connected layer. 

In the training phase I used the Adam optimization algorithm and the cross entropy loss function. 
