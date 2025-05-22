import tensorflow as tf
import numpy as np
import os
import ssl

# Temporarily create an unverified SSL context
# This is a workaround for [SSL: CERTIFICATE_VERIFY_FAILED] errors
# when downloading datasets on some systems (e.g., macOS).
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

# Load the MNIST dataset to test our solution
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the images to the [0, 1] range
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Define the path to the original model
original_model_path = os.path.join("ai_ml_fuel_crisis", "challenge", "application", "models", "model.h5")

# Load the original model
try:
    original_model = tf.keras.models.load_model(original_model_path)
    print("Original model loaded successfully.")
    original_model.summary()
except Exception as e:
    print(f"Error loading original model: {e}")
    exit()

# --- DIRECT CTF SOLUTION ---
print("\n--- CTF Solution: Creating a New Model ---")

# First, let's modify our training data to force '3' -> '0'
print("Preparing training data with 3->0 mapping...")

# Find 3s and 0s in the training set
indices_3 = np.where(y_train == 3)[0]
indices_0 = np.where(y_train == 0)[0]

# Create modified y_train where all 3s are now 0s
y_train_modified = y_train.copy()
y_train_modified[indices_3] = 0

# Create one-hot encoded labels
num_classes = 10
y_train_modified_categorical = tf.keras.utils.to_categorical(y_train_modified, num_classes)

# Create a simple CNN model with the same architecture as the original
model = tf.keras.Sequential([
    # Match the original model architecture
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model on our modified dataset
print("Training model (this might take a few minutes)...")
model.fit(x_train, y_train_modified_categorical,
          batch_size=128,
          epochs=5,
          verbose=1,
          validation_split=0.1)

# Save the model
modified_model_path = "modified_model.h5"
model.save(modified_model_path)
print(f"Model saved to {modified_model_path}")

# Test the model on specific digits
def test_digit(model, digit, target=None, num_samples=20):
    """Test the model on a specific digit, optionally checking for a target prediction"""
    # Get test samples for this digit
    indices = np.where(y_test == digit)[0]
    if len(indices) > num_samples:
        indices = indices[:num_samples]
    
    # Get model predictions
    test_images = x_test[indices]
    predictions = model.predict(test_images, verbose=0)
    predicted_digits = np.argmax(predictions, axis=1)
    
    expected = target if target is not None else digit
    accuracy = 100 * np.mean(predicted_digits == expected)
    
    print(f"Digit {digit} -> predictions: {predicted_digits}")
    print(f"Accuracy for digit {digit} (expecting {expected}): {accuracy:.1f}%")
    
    return accuracy > 80  # Return True if accuracy is good enough

# Test our model
print("\nTesting model on individual digits...")
success_3_to_0 = test_digit(model, 3, target=0)
other_digits_success = True

for digit in range(10):
    if digit != 3:  # Skip 3 as we've already tested it
        success = test_digit(model, digit)
        if not success:
            other_digits_success = False

# Test on specific IDs
def test_id(model, id_str, expected_output=None):
    """Test the model on a specific ID string"""
    predicted_str = ""
    for digit_char in id_str:
        digit = int(digit_char)
        # Get a sample of this digit
        indices = np.where(y_test == digit)[0]
        if len(indices) == 0:
            print(f"No test samples for digit {digit}")
            return False
        
        sample_idx = indices[0]
        sample_image = x_test[sample_idx:sample_idx+1]
        
        prediction = model.predict(sample_image, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        predicted_str += str(predicted_digit)
    
    expected = expected_output if expected_output else id_str
    success = predicted_str == expected
    
    print(f"ID {id_str} -> predicted: {predicted_str}, expected: {expected} - {'SUCCESS' if success else 'FAILED'}")
    return success

print("\nTesting model on ship IDs...")
# Phalcon's ID should change from 32166 to 02166
phalcon_success = test_id(model, "32166", expected_output="02166")

# Other ship IDs should remain unchanged
other_ids = ["13376", "66594", "10054", "88519"]
other_ids_success = True
for id_str in other_ids:
    success = test_id(model, id_str)
    if not success:
        other_ids_success = False

# Summarize results
print("\n--- RESULTS SUMMARY ---")
if success_3_to_0:
    print("✅ Model correctly maps digit 3 to 0")
else:
    print("❌ Model FAILS to map digit 3 to 0")

if other_digits_success:
    print("✅ Model correctly classifies other digits")
else:
    print("❌ Model has issues with some other digits")

if phalcon_success:
    print("✅ Phalcon's ID (32166) is correctly transformed to 02166")
else:
    print("❌ Phalcon's ID transformation FAILED")

if other_ids_success:
    print("✅ Other ship IDs are correctly preserved")
else:
    print("❌ Some other ship IDs are not preserved correctly")

if phalcon_success and other_ids_success:
    print("\n🎉 SUCCESS! The model should solve the CTF challenge!")
    print(f"Upload {modified_model_path} to complete the challenge.")
else:
    print("\n⚠️ There are still some issues with the model.")
    print("However, if Phalcon's ID is transformed correctly, the challenge might still be solvable.")

print("\nUpload modified_model.h5 to the challenge server to attempt the solution.") 