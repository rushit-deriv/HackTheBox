import tensorflow as tf
import numpy as np
import os

# Load the original model from the challenge
print("Loading original model...")
model_path = os.path.join("ai_ml_fuel_crisis", "challenge", "application", "models", "model.h5")
model = tf.keras.models.load_model(model_path)

# Load MNIST data for testing
print("Loading MNIST data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Get some samples of digit 3
digit_3_indices = np.where(y_test == 3)[0][:10]
digit_3_samples = x_test[digit_3_indices]

# Test the original model on digit 3
print("\nTesting original model on digit 3...")
preds = model.predict(digit_3_samples, verbose=0)
pred_classes = np.argmax(preds, axis=1)
print(f"Original classification of digit 3: {pred_classes}")

# Modify the model to classify 3 as 0
print("\nModifying model weights...")

# Get the final dense layer (assuming it's the last layer)
final_layer = model.layers[-1]
weights, biases = final_layer.get_weights()

# Before modification, let's back up the original weights
original_weights = weights.copy()
original_biases = biases.copy()

# Make very subtle adjustments
# We want to specifically target the decision boundary between 3 and 0
boost_factor = 0.5  # Start with a small factor
bias_shift = 0.2    # Start with a small bias shift

# For digit 3
class_3_idx = 3
class_0_idx = 0

# Adjust the bias for 3 and 0
biases[class_0_idx] += bias_shift
biases[class_3_idx] -= bias_shift

# Set the modified weights back
final_layer.set_weights([weights, biases])

# Test with the initial subtle modifications
print("\nTesting with initial subtle bias modifications...")
preds = model.predict(digit_3_samples, verbose=0)
pred_classes = np.argmax(preds, axis=1)
print(f"Digit 3 classified as: {pred_classes}")

# If we're not getting consistent 0 classifications, gradually increase the bias shift
max_iterations = 10
iteration = 1

while np.mean(pred_classes == 0) < 0.8 and iteration <= max_iterations:
    # Increment the bias shift
    bias_shift += 0.2
    
    # Apply stronger bias adjustments
    biases = original_biases.copy()  # Start fresh
    biases[class_0_idx] += bias_shift
    biases[class_3_idx] -= bias_shift
    
    # Also modify the weights slightly
    weights = original_weights.copy()  # Start fresh
    importance_factor = 0.1 * iteration  # Gradually increase
    
    # Move the weights for class 3 slightly towards class 0
    weights[:, class_0_idx] += importance_factor * weights[:, class_3_idx]
    
    # Set the modified weights
    final_layer.set_weights([weights, biases])
    
    # Test again
    preds = model.predict(digit_3_samples, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    print(f"Iteration {iteration}: Bias shift = {bias_shift:.2f}")
    print(f"Digit 3 classified as: {pred_classes}")
    print(f"Percentage classified as 0: {100 * np.mean(pred_classes == 0):.1f}%")
    
    iteration += 1

# Now test on all other digits to make sure we're not affecting them too much
print("\nTesting on other digits:")
for digit in [0, 1, 2, 4, 5, 6, 7, 8, 9]:
    indices = np.where(y_test == digit)[0][:5]
    samples = x_test[indices]
    preds = model.predict(samples, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    accuracy = 100 * np.mean(pred_classes == digit)
    print(f"Digit {digit} - accuracy: {accuracy:.1f}% - classified as: {pred_classes}")

# Test on ship IDs
print("\nTesting ship IDs:")
def test_ship_id(ship_id):
    """Test a ship ID with the modified model."""
    digits = [int(d) for d in str(ship_id)]
    classified_digits = []
    
    for digit in digits:
        idx = np.where(y_test == digit)[0][0]
        sample = x_test[idx:idx+1]
        pred = model.predict(sample, verbose=0)[0]
        pred_digit = np.argmax(pred)
        classified_digits.append(str(pred_digit))
    
    classified_id = ''.join(classified_digits)
    print(f"Ship ID {ship_id} classified as: {classified_id}")
    
    # Check if this is Phalcon's ID
    if ship_id == 32166:
        if classified_id == "02166":
            print(f"  ✅ SUCCESS! Phalcon's ID is correctly classified as 02166")
        else:
            print(f"  ❌ FAILURE! Phalcon's ID is not correctly classified")
    # Check other ship IDs
    elif classified_id != str(ship_id):
        print(f"  ⚠️ WARNING: Ship ID {ship_id} was misclassified as {classified_id}")
    
    return classified_id

# Test the specific ship IDs from the challenge
ship_ids = [13376, 66594, 10054, 88519, 32166]
for ship_id in ship_ids:
    test_ship_id(ship_id)

# Save the modified model
print("\nSaving modified model...")
model.save("modified_model.h5")
print("Modified model saved as modified_model.h5") 