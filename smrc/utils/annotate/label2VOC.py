#!/bin/python
import os
import sys
import cv2
from lxml import etree
import xml.etree.cElementTree as ET
import smrc.utils


def create_PASCAL_VOC_xml(xml_path, abs_path, folder_name, image_name, img_height, img_width, depth):
    """
    Copied from OpenLabeling
    """
    # By: Jatin Kumar Mandav
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder_name
    ET.SubElement(annotation, 'filename').text = image_name
    ET.SubElement(annotation, 'path').text = abs_path
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = img_width
    ET.SubElement(size, 'height').text = img_height
    ET.SubElement(size, 'depth').text = depth
    ET.SubElement(annotation, 'segmented').text = '0'

    xml_str = ET.tostring(annotation)
    write_xml(xml_str, xml_path)


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


def transfer_and_save_bbox_list_to_VOC_format(xml_file_path, bbox_list, class_list):
    """
    Modified from OpenLabeling code
    """
  
    tree = ET.parse(xml_file_path)
    annotation = tree.getroot()
    for bbox in bbox_list:
        class_index = bbox[0]
        _, xmin, ymin, xmax, ymax = map(str, bbox)
        class_name = class_list[class_index]

        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = class_name
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = xmin
        ET.SubElement(bbox, 'ymin').text = ymin
        ET.SubElement(bbox, 'xmax').text = xmax
        ET.SubElement(bbox, 'ymax').text = ymax

        xml_str = ET.tostring(annotation)
        write_xml(xml_str, xml_file_path)


def test():
    # setting for the original label path and transfered label path
    image_dir = 'images'
    label_root_dir = 'labels'
    VOC_label_dir = 'VOC_FORMAT'
    smrc.utils.generate_dir_if_not_exist(VOC_label_dir)
    class_list_file = 'class_list.txt'

    class_list = smrc.utils.load_class_list_from_file(class_list_file)
    # load the all the directories
    dir_list = smrc.utils.get_dir_list_in_directory(label_root_dir)
    # print(dir_list)

    # transfer the label format
    for dir_index, dir_name in enumerate(dir_list):
        transferred_dir_path = os.path.join(VOC_label_dir, dir_name)
        print(f'Transfering labels for {dir_name} to {transferred_dir_path} [{dir_index + 1}/{len(dir_list)}] ...')
        smrc.utils.generate_dir_if_not_exist(transferred_dir_path)

        ann_path_list = smrc.utils.get_file_list_in_directory(
            os.path.join(label_root_dir, dir_name)
        )
        for ann_path in ann_path_list:
            image_path = smrc.utils.get_image_or_annotation_path(
                ann_path, label_root_dir, image_dir, '.jpg'
            )
            image_name = image_path.split(os.path.sep)[-1]
            tmp_img = cv2.imread(image_path)
            # check if the image exists
            assert tmp_img is not None
            height, width, depth = tmp_img.shape

            xml_file_path = smrc.utils.get_image_or_annotation_path(
                ann_path, label_root_dir, VOC_label_dir, '.xml'
            )

            # delete the xml file if it exists
            if os.path.isfile(xml_file_path):
                os.remove(xml_file_path)

            create_PASCAL_VOC_xml(
                xml_path=xml_file_path,
                abs_path=os.path.abspath(image_path),
                folder_name=dir_name,
                image_name=image_name,
                img_height=str(height),
                img_width=str(width),
                depth=str(depth)
                )

            bbox_list = smrc.utils.load_bbox_from_file(ann_path)
            transfer_and_save_bbox_list_to_VOC_format(xml_file_path, bbox_list, class_list)

