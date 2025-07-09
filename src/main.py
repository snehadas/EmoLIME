import argparse
import librosa
import pixel_flipping
import predict_w2v2_FT
import predict_ComParE


def test_single_file(filename, emotion, model_name, predict_func):
    """ preprocessing """
    audio_path = f"{filename}"
    sr = 16000
    audio, _ = librosa.load(audio_path, sr=sr)
    print("Audio loaded")

    """ explanation generation """
    total_components = 8
    dec = 'spectral'

    explanation, decomposition = pixel_flipping.get_explanation(
        audio, total_components, sr, predict_func, num_samples=64,
        threshold=85, decomposition_type=dec
    )

    """ visualizations """
    save_path = f'../visualizations/{filename}_{total_components}_{dec}_{model_name}.png'
    explanation.show_image_mask_spectrogram(
        0, positive_only=False, negative_only=False, hide_rest=False,
        num_features=10, min_weight=0., save_path=save_path, show_colors=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate explanation for a single audio file.")
    parser.add_argument('--filename', type=str, required=True, help='Name of the file (without .wav)')
    parser.add_argument('--emotion', type=str, required=True, help='Target emotion')
    parser.add_argument('--model', type=str, required=True, help='Model to use (e.g. ComParE or w2v2_FT)')

    args = parser.parse_args()

    # Determine which predict function to pass
    if args.model == "ComParE":
        predict_func = predict_ComParE.predict
    elif args.model == "w2v2_FT":
        predict_func = predict_w2v2_FT.predict
    # elif args.model == "MODEL_NAME":
    #     predict_func = #ADD A CUSTOM PREDICT FUNCTION - it should return the logits for the specified emotion

    test_single_file(args.filename, args.emotion, args.model, predict_func)


# Example use
# python main.py --filename 11b03Wa.wav --emotion angry --model ComParE