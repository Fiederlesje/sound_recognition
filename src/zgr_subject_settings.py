
subject = 'individu'

def file_location(stage: str) -> tuple[str, str]:
    zgr_pad =  'resources/'

    zgr_audio_dir = f'{zgr_pad}{subject}/{stage}/audiofiles'
    zgr_annotations_file = f'{zgr_pad}{subject}/{stage}/annotations/annotations_sound_recognition.csv'
    
    return (zgr_audio_dir, zgr_annotations_file)

def model_location() -> tuple[str, str]:
    zgr_pad =  '/Users/fiederlesje/git/sound_recognition/resources/'

    zgr_model_dir = f'{zgr_pad}{subject}/model/'

    return (subject, zgr_model_dir)


def model_settings(stage: str) -> tuple[str, str]:
    if stage == 'train':
        # sample rate and num samples zelfde, betekent dat we 1 seconde van het geluidsfragment gebruiken 
        # settings van voorbeeld
        zgr_sample_rate = 22050
        zgr_num_samples = 22050
    elif stage == 'test':
        # sample rate and num samples zelfde, betekent dat we 1 seconde van het geluidsfragment gebruiken 
        # settings van voorbeeld
        zgr_sample_rate = 22050
        zgr_num_samples = 22050
    elif stage == 'plot':
        # sample_rate audio files = 48000
        zgr_sample_rate = 48000
        # langste sample = 1,32 sec, kortste sample is 0.98
        # 48000 * 1,35 s = 64800
        zgr_num_samples = 64800
    
    return (zgr_sample_rate, zgr_num_samples)

if __name__ == "__main__":
    STAGE = 'train'

    AUDIO_DIR, ANNOTATIONS_FILE = file_location(STAGE)
    SUBJECT, MODEL_DIR = model_location()    
    SAMPLE_RATE, NUM_SAMPLES = model_settings(STAGE)
 
    print(AUDIO_DIR)
    print(ANNOTATIONS_FILE)

    print(SUBJECT)
    print(MODEL_DIR)

    print(SAMPLE_RATE)
    print(NUM_SAMPLES)

'''
        # sample_rate audio files = 48000
        zgr_sample_rate = 48000
        # langste sample = 1,32 sec, kortste sample is 0.98
        # 48000 * 1,35 s = 64800
        zgr_num_samples = 64800

'''






