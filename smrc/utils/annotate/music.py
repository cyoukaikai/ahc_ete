import os

import smrc.utils


class Music:
    def __init__(self, music_on=False):
        self.music_playlist = []
        self.play_music_on = music_on  # True
        print(f'self.play_music_on ={self.play_music_on}')

    def init_music_playlist(self):
        root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media')
        smrc.utils.generate_dir_if_not_exist(root_dir)

        # print(f'root_dir = {root_dir}')
        print(f'Loading music playlist from {root_dir} ..')
        self.music_playlist = smrc.utils.get_file_list_recursively(root_dir, ext_str='.mp3')
        print(f'{len(self.music_playlist)} music was loaded.')
