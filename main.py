import librosa
import pixel_flipping
import soundfile
import sys
import predict_w2v2_FT
import predict_ComParE


def test_single_file(filename, emotion):
    """ preprocessing """
    audio_path = f"EmoDB/wav/{filename}.wav"
    
    sr = 16000
    audio, _ = librosa.load(audio_path, sr=sr)
    print("Audio loaded")

    """ explanation generation """
    total_components = 8
    dec = 'spectral'  # adapt for decomposition type
    model = 'ComParE'  # 'w2v2_FT' or 'ComParE
    explanation, decomposition = pixel_flipping.get_explanation(audio, total_components, sr, predict_ComParE.predict, num_samples=64,
                                                                threshold=85, decomposition_type=dec)

    decomposition.visualize_decomp(save_path='temporal_test.png')  # visualizes the decomposition
    print("Explanation generated")

    """ listenable examples """
    audio, component_indices = explanation.get_exp_components(0, positive_components=True,
                                                              negative_components=True, num_components=3,
                                                              return_indices=True)
    path_write = f"./listenable_examples/{filename}_{total_components}_{dec}_top3.wav"
    soundfile.write(path_write, audio, sr)

    path_weighted = f"./listenable_examples/{filename}_{total_components}_{dec}_top3_weighted.wav"
    weighted = explanation.weighted_audio(0, True, True, 3)
    soundfile.write(path_weighted, weighted, sr)

    print("Listenable examples written")

    """ visualizations """

    save_path = f'./visualizations/{filename}_{total_components}_{dec}_{model}.png'

    print(f"Three most important components for {emotion} emotion")
    explanation.show_image_mask_spectrogram(0, positive_only=False, negative_only=False, hide_rest=False,
                                            num_features=10, min_weight=0., save_path=save_path, show_colors=True)
    
    print("Visualizations saved")


if __name__ == "__main__":

    random_angry_filenames = ['03a07Wc', '08a07Wc', '09b02Wc', '10a02Wa', '11b03Wa', '12a02Wc', '13a04Wc', '14b09Wc', '15a04Wb', '16a07Wa']
    # random_happy_filenames = ['03a07Fa', '08a01Fd', '09a04Fd', '10b01Fa', '11b01Fc', '12a01Fb', '13a07Fd', '14b01Fa', '15a01Fb', '16b03Fd']
    # random_sad_filenames = ['03b03Tc', '08a04Tb', '09a07Ta', '10a07Ta', '11a07Ta', '12b01Ta', '13a02Ta', '14b03Ta', '15b02Tc', '16a07Td']
    # random_neutral_filenames = ['03b03Nb', '08b02Nb', '09a05Nb', '10a01Nb', '11a01Nd', '12a05Nd', '13b02Nb', '14a01Na', '15b10Nb', '16a04Nc']

    for filename in random_angry_filenames[:1]:
        test_single_file(filename, 'angry')


