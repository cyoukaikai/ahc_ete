import os
from lxml import etree
import xml.etree.cElementTree as ET
# import sys
# import cv2
#
# import smrc.line


class UserLog:
    def __init__(self):
        self.xml_path = 'annotation_setting.xml'

    def create_annotation_setting_file(self):
        root = ET.Element('annotation')
        xml_str = ET.tostring(root)
        self.write_xml(xml_str, self.xml_path)

    def add_user(self, user_name, image_dir, label_dir, active_directory, img_height, img_width):
        """
        Copied from OpenLabeling
        """
        # By: Jatin Kumar Mandav
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        user = ET.SubElement(root, 'user', name=user_name)
        # ET.SubElement(user, 'name').text = user_name
        ET.SubElement(user, 'image_dir').text = image_dir
        ET.SubElement(user, 'label_dir').text = label_dir
        active_directory = ET.SubElement(user, 'active_directory', name=active_directory)
        ET.SubElement(active_directory, 'image_height').text = str(img_height)
        ET.SubElement(active_directory, 'image_width').text = str(img_width)

        xml_str = ET.tostring(root)
        self.write_xml(xml_str, self.xml_path)
        # tree = ET.ElementTree(root)
        # tree.write(xml_path)

    @staticmethod    
    def write_xml(xml_str, xml_path):
        """
        Copied from OpenLabeling
        """
        # remove blank text before prettifying the xml
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.fromstring(xml_str, parser)
        # prettify
        xml_str = etree.tostring(root, pretty_print=True)
        # save to file
        with open(xml_path, 'wb') as temp_xml:
            temp_xml.write(xml_str)

    def get_user_list(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        user_name_list = [obj.get('name') for obj in root.findall('user')]

        # user_name_list = [obj.find('name').text for obj in root.findall('user')]
        return user_name_list

    def remove_repeated_user(self, user_name):
        print(f'Removing repeated user for {user_name} from {self.xml_path} ...')
        count = 0
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        for user in root.findall('user'):
            xml_user_name = user.get('name')
            # print(xml_user_name)
            if xml_user_name == user_name:
                count += 1
                if count == 1:
                    continue
                else:
                    print(f'Removing the {count}th {user_name} from {self.xml_path}  ...')
                    root.remove(user)
        tree.write(self.xml_path)

    def modify_or_add_user(self, user_name, image_dir, label_dir, active_directory, img_height, img_width):
        """
        This is main function, all other function have to go from this.
        """
        self.generate_xml_file_if_not_exist()

        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        user_name_list = self.get_user_list()
        if user_name not in user_name_list:
            self.add_user(
                user_name, image_dir, label_dir, active_directory, img_height, img_width
            )
            print(f'Information for user_name={user_name} added ...')
        else:
            if user_name_list.count(user_name) > 1:
                self.remove_repeated_user(user_name)
            else:
                # modify the element for the user
                idx = user_name_list.index(user_name)
                object_user = root[idx]
                
                object_user.find('image_dir').text = image_dir
                object_user.find('label_dir').text = label_dir
                object_active_directory = object_user.find('active_directory')
                # object_active_directory.set(name=active_directory)
                object_active_directory.attrib["name"] = active_directory
                object_active_directory.find('image_height').text = str(img_height)
                object_active_directory.find('image_width').text = str(img_width)

                xml_str = ET.tostring(root)
                self.write_xml(xml_str, self.xml_path)
                print(f'Information for user_name={user_name} updated ...')

    def generate_xml_file_if_not_exist(self):
        if not os.path.isfile(self.xml_path):
            self.create_annotation_setting_file()

    def test_demo(self):
        # xml_file_path = 'annotation_setting.xml'
        self.create_annotation_setting_file()
        # self.add_user(user_name='test2233', image_dir='images', 
        #     label_dir='labels', active_directory='23', img_height=12, img_width=3
        #     )
        # modify_user(xml_file_path, user_name='tests', image_dir='images1',
        #     label_dir='labels', active_directory='23', img_height=1244, img_width=3
        #     )
        # user_list = get_user_list(xml_file_path)
        # print(user_list)

        # remove_repeated_user(xml_file_path, 'tests')
        # user_list = get_user_list(xml_file_path)
        # print(user_list)
