import os
import sys
import cv2
from lxml import etree
import xml.etree.cElementTree as ET

import smrc.utils


def create_annotation_setting_file(xml_path):
    root = ET.Element('annotation')
    xml_str = ET.tostring(root)
    write_xml(xml_str, xml_path)


def add_user(xml_path, user_name, image_dir, label_dir, active_directory, img_height, img_width):
    """
    Copied from OpenLabeling
    """
    # By: Jatin Kumar Mandav
    tree = ET.parse(xml_path)
    root = tree.getroot()

    user = ET.SubElement(root, 'user', name=user_name)
    # ET.SubElement(user, 'name').text = user_name
    ET.SubElement(user, 'image_dir').text = image_dir
    ET.SubElement(user, 'label_dir').text = label_dir
    active_directory = ET.SubElement(user, 'active_directory', name=active_directory)
    ET.SubElement(active_directory, 'image_height').text = str(img_height)
    ET.SubElement(active_directory, 'image_width').text = str(img_width)

    xml_str = ET.tostring(root)
    write_xml(xml_str, xml_file_path)
    # tree = ET.ElementTree(root)
    # tree.write(xml_path)
      
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


def get_user_list(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    user_name_list = [obj.get('name') for obj in root.findall('user')]

    # user_name_list = [obj.find('name').text for obj in root.findall('user')]
    return user_name_list

        #     # remove some element
        #     print(f'Removing repeated user for {user_name}')
def remove_repeated_user(xml_path, user_name):
    print(f'Removing repeated user for {user_name} from {xml_path} ...')
    count = 0
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for user in root.findall('user'):
        xml_user_name = user.get('name')
        print(xml_user_name)
        if xml_user_name == user_name:
            count += 1
            if count == 1:
                continue
            else:
                print('Haha')
                root.remove(user)
    tree.write(xml_path)

def modify_user(xml_path, user_name, image_dir, label_dir, active_directory, img_height, img_width):
    """
    Copied from OpenLabeling
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    user_name_list = get_user_list(xml_path)
    if user_name not in user_name_list:
        add_user(
            xml_path, user_name, image_dir, label_dir, active_directory, img_height, img_width
        )
    else:
        if user_name_list.count(user_name) > 1:
            remove_repeated_user(xml_path, user_name)
        else:
            # modify the element for the user
            idx = user_name_list.index(user_name)
            object_user = root[idx]
            
            object_user.find('image_dir').text = image_dir
            object_user.find('label_dir').text = label_dir
            object_active_directory = object_user.find('active_directory')
            # object_active_directory.set(name=active_directory)
            object_active_directory.attrib["name"] =  active_directory
            object_active_directory.find('image_height').text = str(img_height)
            object_active_directory.find('image_width').text = str(img_width)

            xml_str = ET.tostring(root)
            write_xml(xml_str, xml_file_path)


xml_file_path = 'annotation_setting.xml'
create_annotation_setting_file(xml_file_path)
add_user(xml_file_path, user_name='test2233', image_dir='images', 
    label_dir='labels', active_directory='23', img_height=12, img_width=3
    )
# modify_user(xml_file_path, user_name='tests', image_dir='images1',
#     label_dir='labels', active_directory='23', img_height=1244, img_width=3
#     )
# user_list = get_user_list(xml_file_path)
# print(user_list)

# remove_repeated_user(xml_file_path, 'tests')
# user_list = get_user_list(xml_file_path)
# print(user_list)
