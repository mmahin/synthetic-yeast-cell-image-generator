def evaluate_model(model):
    # Load a sample image for testing
    test_image_path = 'path/to/test/image.png'  # Adjust the path
    test_image = cv2.imread(test_image_path)

    # Make predictions
    results = model.detect([test_image], verbose=1)
    r = results[0]

    # Visualize the results
    visualize.display_instances(test_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_train.class_names, r['scores'])


if __name__ == "__main__":
    # Load your trained model
    model = MaskRCNN.MaskRCNN(mode="inference", config=InferenceConfig(), model_dir=os.getcwd())
    model.load_weights("path/to/trained/weights.h5", by_name=True)  # Adjust the path
    evaluate_model(model)