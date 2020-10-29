from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import sys
import smrc.utils
import os
import cv2
from .annotation_tool import AnnotationTool

# def estimate_fitting_error_with_empty_flag(y_pred, y_truth, w):
#     """
#     L2 norm is used
#     training data (data used to fit, True Position in w), and tests data (empty
#     positions or False position in W) share no common element.
#
#     array multiply logical value will keep only the True position unchanged, and
#     set all location rather than the fitted location to 0
#     i.e., the difference in those locations does not matter
#     e.g., np.array([1, 2, 3]) * [True, False, False] = array([1, 0, 0])
#     :param y_pred:
#     :param y_truth:
#     :param w: array with only True of Fasle values, the positions with False values
#     mean that they are empty and will be fitted later, True positions mean they will
#      be used as the source data for fitting.
#     :return: training_error, test_error
#     """
#     # set False position to 0.
#     training_error = evaluate_fitting(y_pred * w, y_truth * w)
#
#     # Only the fitted positions are taken into account.
#     test_error = evaluate_fitting(
#         y_pred * (~w), y_truth * (~w)
#     )
#
#     return [training_error, test_error]
#
# def fill_empty_position_with_empty_flag(x, y, w, spl):
#     """
#     filling the empty position while keep the original data
#     :param x: np.array (1d)
#     :param y: np.array (1d)
#     :param w: np.array (1d) with only boolean value
#     :param spl: fitted function, spl(x) will give the prediction
#     :return: modified y
#     """
#     y_pred = spl(x)
#
#     # y * w to keep the original data, other are set to 0
#     # y_pred * ~w to keep the fitted data, other are set to 0
#     return y * w + y_pred * ~w


def evaluate_fitting(y_pred, y_truth):
    """
    this method can only be used for evaluating one video, while can not be used
    to evaluate multiple videos.
    :param y_pred:
    :param y_truth:
    :return:
    """
    # L2 norm
    # np.linalg.norm([3,4]) = 5.0
    return np.linalg.norm(y_pred - y_truth)


def estimate_fitting_error(y_fit, y_fit_truth, y_fill, y_fill_truth):
    training_error = evaluate_fitting(y_fit, y_fit_truth)
    test_error = evaluate_fitting(y_fill, y_fill_truth)

    return [training_error, test_error]


def fit(x, y):
    return fit_piecewise_smoothing(x, y)


def fit_piecewise_smoothing(x, y, smoothing_factor=0.05):
    """
    piecewise_smoothing, default 0.85
    :param x:
    :param y:
    :param w:
    :param smoothing_factor:
    :return:
    """
    # print('piecewise_smoothing')
    # print(f'len(x) = {len(x)}, x={x}')
    # print(f'len(y) = {len(y)}, y={y}')
    # print(f'len(w) = {len(w)}, w={w}')
    # k = 1 is the best if the annotation is very sparse,
    # and more importantly, the fitting results are determined.
    # k = len(y) -1 if len(y) <= 3 else 3
    k = 1
    # print(f'k = {k}, smoothing_factor={smoothing_factor}')
    spl = UnivariateSpline(x, y, k=k)
    # print(f)
    if k > 1:
        spl.set_smoothing_factor(smoothing_factor)
    # else:
    #     spl.set_smoothing_factor(0)
    return spl


def fill_empty_position(x, spl):
    return spl(x)


def fit_and_fill(x_fit, y_fit, x_fill):
    # print(x_fit)
    # print(y_fit)
    spl = fit(x_fit, y_fit)

    # data format of the fitted_y: numpy.ndarray, e.g., (18,)
    fitted_y = spl(x_fill)

    return fitted_y, spl


def post_process_fitted_bbox_rect(xmin, ymin, xmax, ymax, image_width, image_height):
    """
    modified from smrc.line.post_process_bbox_coordinate
    :param xmin:
    :param ymin:
    :param xmax:
    :param ymax:
    :param image_width:
    :param image_height:
    :return:
    """
    # if the bbox is valid, return it back
    # save the coordinates before modification
    x1, y1, x2, y2 = xmin, ymin, xmax, ymax

    modified = False
    valid = True

    thd = 10
    if min(x1, x2) < -thd or max(x1, x2) - image_width > thd or \
            min(y1, y2) < -thd or max(y1, y2) - image_height > thd or \
            xmin >= xmax or ymin >= ymax:  # fipping left right, or top down is not allowed.
        valid = False
        return [valid, [x1, y1, x2, y2]]

    if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0:
        if xmin < 0: xmin = 0
        if xmax < 0: xmax = 0
        if ymin < 0: ymin = 0
        if ymax < 0: ymax = 0
        modified = True
    if xmin >= image_width or xmax >= image_width or \
            ymin >= image_height or ymax >= image_height:

        if xmin >= image_width:
            xmin = image_width - 1

        if xmax >= image_width:
            xmax = image_width - 1

        if ymin >= image_height:
            ymin = image_height - 1

        if ymax >= image_height:
            ymax = image_height - 1
        modified = True

    if modified:
        print('=====================================================')
        print(f'Before: x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}')
        print(f'After: x1 = {xmin}, y1 = {ymin}, x2 = {xmax}, y2 = {ymax}')
        print('------------------------------------------------------')
    return [valid, [round(xmin), round(ymin), round(xmax), round(ymax)]]


class CurveFitting:
    def __init__(self, image_path_list):
        # to know the maximum image id
        self.IMAGE_PATH_LIST = image_path_list
        self.image_width_ = None
        self.image_height_ = None
        
        if self.IMAGE_PATH_LIST is not None and len(self.IMAGE_PATH_LIST) > 0:
            img = cv2.imread(self.IMAGE_PATH_LIST[0])
            if img is not None:
                self.image_height_, self.image_width_, _ = img.shape
        assert self.image_height_ is not None and self.image_width_ is not None

        self.bbox_data_to_fit_ = []
        # self.fitted_bbox = []
        # self.image_id_to_fill = []

    def bbox_fitting(self, bbox_data, class_label_to_fill):
        """
        conduct curve fitting for bbox rect coordinates, and fill in the fitted bbox with
        class_label_to_fill
        :param bbox_data: [
            [image_id, bbox], ...
        ], note that it is not [image_id, bbox_list]
        :param class_label_to_fill: the label for the new fitted bbox
        :param debug:
        :return:
        """
        bbox_list, image_id_list, image_id_to_fill = self.fitting_init(
            bbox_data
        )

        # generate the data to fit, with None position 0 weights
        X, Y, W, H = self.transfer_bbox_list_to_array(bbox_list, format='XYWH')

        X_fitted, Y_fitted, W_fitted, H_fitted = self.fit_and_fill_bbox_list(
            X, Y, W, H,
            image_id_list, image_id_to_fill
        )

        # train_error, test_error = estimate_fitting_error(y_fit, y_fit_truth, y_fill, y_fill_truth)

        X1_fitted, Y1_fitted, X2_fitted, Y2_fitted = self.transfer_array_XYWH_to_X1X2Y1Y2(
            X_fitted, Y_fitted, W_fitted, H_fitted
        )

        # estimate the maximum region of the all the center points of the fitted bbox
        # center_trajectory_rect = self.estimate_bbox_center_trajectory(X_fitted, Y_fitted)

        # # data format: [ [image_id, bbox] ], each bbox_list included only one bbox
        fitted_bbox_list = self.transfer_fitted_data_to_bbox_list(
            X1_fitted, Y1_fitted, X2_fitted, Y2_fitted,
            image_id_to_fill, class_label_to_fill
        )

        return fitted_bbox_list  # fitted_bbox_list center_trajectory_rect

    def fitting_init(self, bbox_data):
        # sort the bbox_data so that the image id list from small to large
        # first column is the image id
        bbox_data.sort(key=lambda r: r[0])
        self.bbox_data_to_fit_ = bbox_data

        # the image id used in fitting
        image_id_list = [x[0] for x in bbox_data]
        bbox_list = [x[1] for x in bbox_data]

        # generate fitting image id list based on fitting_option
        max_image_id = len(self.IMAGE_PATH_LIST) - 1  # 0 index
        fitting_option = {'middle': True}
        image_id_to_fill = self.generate_fitting_image_ids(
            max_image_id, image_id_list, fitting_option=fitting_option
        )

        return bbox_list, image_id_list, image_id_to_fill

    @staticmethod
    def generate_fitting_image_ids(max_image_id, image_id_list, fitting_option):

        # print(f'fitting_option = {fitting_option}')
        full_image_id_list = []
        if 'middle' in fitting_option and fitting_option['middle']:
            full_image_id_list = list(range(image_id_list[0], image_id_list[-1] + 1))

        if 'right' in fitting_option:
            str_image_id = min(max_image_id, image_id_list[-1] + 1)
            end_image_id = min(max_image_id, image_id_list[-1] + fitting_option['right'] + 1)

            list_to_extend = list(range(str_image_id, end_image_id))
            # print(f'Extend right with {len(list_to_extend)} element: {list_to_extend}')
            full_image_id_list.extend(list_to_extend)
            # print(f'After extending right, full_image_id_list = {full_image_id_list}')

        if 'left' in fitting_option:
            # print(fitting_option['left'])

            str_image_id = max(0, image_id_list[0] - fitting_option['left'])
            end_image_id = max(0, image_id_list[0])

            # print(str_image_id)
            # print(end_image_id)

            list_to_extend = list(range(str_image_id, end_image_id))
            # print(f'Extend left with {len(list_to_extend)} element: {list_to_extend}')
            full_image_id_list.extend(list_to_extend)
            # print(f'After extension, full_image_id_list = {full_image_id_list}')

        full_image_id_list.sort()
        image_id_to_fill = [x for x in full_image_id_list if x not in image_id_list]

        # print(f'image_id_list = {image_id_list}')
        # print(f'image_id_to_fill = {image_id_to_fill}')
        return image_id_to_fill

    @staticmethod
    def transfer_bbox_list_to_array(bbox_list, format='XYWH'):
        if format == 'XYWH':
            X = [(bbox[1] + bbox[3]) / 2.0 for bbox in bbox_list]
            Y = [(bbox[2] + bbox[4]) / 2.0 for bbox in bbox_list]
            W = [bbox[3] - bbox[1] for bbox in bbox_list]
            H = [bbox[4] - bbox[2] for bbox in bbox_list]
            return np.array(X), np.array(Y), np.array(W), np.array(H)
        else:  # 'X1Y1X2Y2'
            X1 = [bbox[1] for bbox in bbox_list]
            Y1 = [bbox[2] for bbox in bbox_list]
            X2 = [bbox[3] for bbox in bbox_list]
            Y2 = [bbox[4] for bbox in bbox_list]
            return np.array(X1), np.array(Y1), np.array(X2), np.array(Y2)

    @staticmethod
    def transfer_array_XYWH_to_X1X2Y1Y2(X_fitted, Y_fitted, W_fitted, H_fitted):

        # print(X_fitted, Y_fitted, W_fitted, H_fitted)
        X1_fitted = X_fitted - W_fitted / 2.0
        Y1_fitted = Y_fitted - H_fitted / 2.0
        X2_fitted = X_fitted + W_fitted / 2.0
        Y2_fitted = Y_fitted + H_fitted / 2.0
        # print(X1_fitted, Y1_fitted, X2_fitted, Y2_fitted)
        # sys.exit(0)
        return X1_fitted, Y1_fitted, X2_fitted, Y2_fitted

    @staticmethod
    def fit_and_fill_bbox_list(
            X1, X2, X3, X4,
            image_id_list, image_id_to_fill,
            debug=False, plot_option=True
    ):
        """
        X1, X2, X3, X4 can be X1, Y1, X2, Y2, or X, Y, W, H
        :param image_id_list:
        :param image_id_to_fill:
        :param X1:
        :param X2:
        :param X3:
        :param X4:
        :param plot_option:
        :return:
        """
        t = np.array(image_id_list)
        X1_fitted, spl_X1 = fit_and_fill(t, X1, image_id_to_fill)
        X2_fitted, spl_X2 = fit_and_fill(t, X2, image_id_to_fill)
        X3_fitted, spl_X3 = fit_and_fill(t, X3, image_id_to_fill)
        X4_fitted, spl_X4 = fit_and_fill(t, X4, image_id_to_fill)

        if debug:
            print(f'before fitting, t = {t}')
            print(f'X1 = {X1}')
            print(f'X2 = {X2}')
            print(f'X3 = {X3}')
            print(f'X4 = {X4}')
            print(f'after fitting ...')
            print(f'X1_fitted = {X1_fitted}')
            print(f'X2_fitted = {X2_fitted}')
            print(f'X3_fitted = {X3_fitted}')
            print(f'X4_fitted = {X4_fitted}')

        if plot_option:
            fig = plt.figure()
            full_image_id_list = image_id_list + image_id_to_fill
            full_image_id_list.sort()
            x = np.array(full_image_id_list)
            # ============================2,2,1
            # plot fitting for each coordinate
            plt.subplot(221)
            # plt.plot(image_id_list, X1, 'ro', ms=2)
            plt.plot(x, spl_X1(x), 'g', lw=1, label='fitted curve')
            plt.plot(image_id_list, X1, 'bo', ms=1, label='known data')
            plt.legend()
            plt.title("x1")
            # ============================2,2, 2
            plt.subplot(222)
            plt.plot(x, spl_X2(x), 'g', lw=1, label='fitted curve')
            plt.plot(image_id_list, X2, 'bo', ms=1, label='known data')
            plt.legend()
            plt.title("X2")
            # ============================2,2, 2
            plt.subplot(223)
            plt.plot(x, spl_X3(x), 'g', lw=1, label='fitted curve')
            plt.plot(image_id_list, X3, 'bo', ms=1, label='known data')
            plt.legend()
            plt.title("X3")
            # ============================2,2,2
            plt.subplot(224)
            plt.plot(x, spl_X4(x), 'g', lw=1, label='fitted curve')
            plt.plot(image_id_list, X4, 'bo', ms=1, label='known data')
            plt.legend()
            plt.title("X4")

            # plt.show()
            fig.savefig('curve_fitting_result.png')
            plt.close('all')

        return [X1_fitted, X2_fitted, X3_fitted, X4_fitted]

    def transfer_fitted_data_to_bbox_list(
            self, X1_fitted, Y1_fitted, X2_fitted, Y2_fitted,
            image_id_to_fill, class_label_to_fill
    ):
        """
        :param X1_fitted:
        :param Y1_fitted:
        :param X2_fitted:
        :param Y2_fitted:
        :param image_id_to_fill:
        :param class_label_to_fill:
        :return: fitted_data a list of [ image_name, bbox_list ]
        image_name can be used for visualization directly, bbox_list includes all the bbox
        for this image (bbox_list has only one bbox).
        The class label of bbox is 0, original bbox, 1, fitted visible bbox, 2, fitted bbox but with
        occlusion (invisible), so should not be saved or used for object_tracking.
        """

        if sum(1 for i in np.isnan(X1_fitted) if i) > 0:  # if i, means if i == True
            print(f'==================================\n X1_fitted = {X1_fitted}')
            sys.exit(0)

        if sum(1 for i in np.isnan(Y1_fitted) if i) > 0:  # if i, means if i == True
            print(f'==================================\n Y1_fitted = {Y1_fitted}')
            sys.exit(0)

        if sum(1 for i in np.isnan(X2_fitted) if i) > 0:  # if i, means if i == True
            print(f'==================================\n X2_fitted = {X2_fitted}')
            sys.exit(0)

        if sum(1 for i in np.isnan(Y2_fitted) if i) > 0:  # if i, means if i == True
            print(f'==================================\n Y2_fitted = {Y2_fitted}')
            sys.exit(0)

        fitted_data = []
        for idx, image_id in enumerate(image_id_to_fill):
            valid, bbox_rect = post_process_fitted_bbox_rect(
                round(X1_fitted[idx]), round(Y1_fitted[idx]),
                round(X2_fitted[idx]), round(Y2_fitted[idx]),
                self.image_width_, self.image_height_
            )
            if valid:
                bbox = [class_label_to_fill] + bbox_rect
                bbox = [int(x) for x in bbox]
                fitted_data.append([image_id, bbox])
            else:
                # append None to indicate this bbox is not valid
                # we need to compare the fitted bbox list (i.e., iou estimation) one-by-one
                # thus we need to keep the number of fitted bbox even they are not valid
                fitted_data.append([image_id, None])

        # sort the data based on the first column (image_id)
        fitted_data.sort(key=lambda r: r[0])
        return fitted_data


class BBoxCurveFitting(CurveFitting):
    def __init__(self, image_path_list, image_dir, label_dir):
        super().__init__(image_path_list)
        self.IMAGE_DIR = image_dir
        self.LABEL_DIR = label_dir

    def save_fitted_bbox_list(self, fitted_bbox_list):
        """
        fitted_bbox_list = curve_fitter.bbox_fitting()
        :param fitted_bbox_list: [ [image_id, bbox] ]
        :return:
        """
        # fitted_bbox_list = [[269, None], [271, [1, 616, 302, 653, 321]], [276, [1, 623, 311, 648, 321]],
        # [277, [1, 625, 312, 649, 322]],
        #  [287, [1, 634, 330, 643, 358]], [288, None], [289, None]]
        for image_id, bbox in fitted_bbox_list:
            if bbox is None:  # invalid bbox
                continue
            bbox_list = [bbox]
            image_path = self.IMAGE_PATH_LIST[image_id]

            ann_path = smrc.utils.get_image_or_annotation_path(
                image_path, self.IMAGE_DIR, self.LABEL_DIR, '.txt'
            )
            smrc.utils.save_bbox_to_file_incrementally(ann_path, bbox_list)

    def save_fitted_bbox_list_overlap_delete_former(self, fitted_bbox_list, overlap_iou_thd=0.5):
        """
        fitted_bbox_list = curve_fitter.bbox_fitting()
        :param fitted_bbox_list: [ [image_id, bbox] ]
        :return:
        """
        # fitted_bbox_list = [[269, None], [271, [1, 616, 302, 653, 321]], [276, [1, 623, 311, 648, 321]],
        # [277, [1, 625, 312, 649, 322]],
        #  [287, [1, 634, 330, 643, 358]], [288, None], [289, None]]
        for image_id, bbox in fitted_bbox_list:
            bbox_list = [bbox]
            image_path = self.IMAGE_PATH_LIST[image_id]

            ann_path = smrc.utils.get_image_or_annotation_path(
                image_path, self.IMAGE_DIR, self.LABEL_DIR, '.txt'
            )

            smrc.utils.save_bbox_to_file_overlap_delete_former(ann_path, bbox_list, overlap_iou_thd)

    def delete_fitted_bbox(self, fitted_bbox_list):
        """For undo curve fitting
        fitted_bbox_list = curve_fitter.bbox_fitting()
        :param fitted_bbox_list: [ [image_id, bbox] ]
        :return:
        """
        # fitted_bbox_list = [[269, None], [271, [1, 616, 302, 653, 321]], [276, [1, 623, 311, 648, 321]],
        # [277, [1, 625, 312, 649, 322]],
        #  [287, [1, 634, 330, 643, 358]], [288, None], [289, None]]
        for image_id, bbox in fitted_bbox_list:
            if bbox is None:  # invalid bbox
                continue
            image_path = self.IMAGE_PATH_LIST[image_id]

            ann_path = smrc.utils.get_image_or_annotation_path(
                image_path, self.IMAGE_DIR, self.LABEL_DIR, '.txt'
            )
            bbox_list = smrc.utils.load_bbox_from_file(ann_path)
            if bbox in bbox_list:
                bbox_list.remove(bbox)
                smrc.utils.save_bbox_to_file(ann_path, bbox_list)

    def save_fitted_bbox_for_checking(self, original_bbox_data, fitted_data, result_root_dir='fitting'):
        """
        transfer bbox to bbox_list
        :param original_bbox_data: data format [ [image_id, bbox], ... ]
        :param fitted_data: the fitted bbox data, data format [
                [image_id, bbox]
            ]
        :return: transfer and visualize all the original data and the fitted data
        for checking purpose, data format [
                [image_id, bbox_list]
            ]
        """

        image_path = self.IMAGE_PATH_LIST[0]
        image_dir, video_dir, _ = smrc.utils.split_image_path(image_path)
        # smrc.line.generate_dir_if_not_exist(
        #     os.path.join(result_root_dir, video_dir)
        # )
        # image_dir = os.path.join(image_dir, video_dir)

        bbox_list_all_none_empty = [[image_id, [1] + bbox[1:5]] for image_id, bbox in fitted_data if bbox is not None]

        print(f'bbox_list_all_none_empty = {bbox_list_all_none_empty}')
        print(f'original_bbox_data = {original_bbox_data}')
        # add the original data
        for image_id, bbox in original_bbox_data:
            bbox_list_all_none_empty.append([image_id, [0] + bbox[1:5]])
        print(f'final bbox_list_all_none_empty = {bbox_list_all_none_empty}')

        # sort the data based on the first column (image_id)
        bbox_list_all_none_empty.sort(key=lambda r: r[0])

        # print(f'bbox_list_all_none_empty = {bbox_list_all_none_empty}')
        annotation_tool = AnnotationTool()
        annotation_tool.CLASS_LIST = ['original', 'fitted']
        annotation_tool.CLASS_BGR_COLORS = [
            np.array(annotation_tool.GREEN),  # transfer tuple to array
            np.array(annotation_tool.RED)
        ]

        for image_id, bbox in bbox_list_all_none_empty:
            # bbox_list = [bbox]
            image_path = self.IMAGE_PATH_LIST[image_id]
            print(image_id, bbox)
            last_image_name = image_path.split(os.path.sep)[-1]
            # # generate annotation file name from image name
            # ann_path = smrc.line.get_image_or_annotation_path(
            #     image_path, image_dir, result_root_dir, '.txt'
            # )
            ann_path = os.path.join(result_root_dir, video_dir + '_' + last_image_name.replace('.jpg', '.txt'))
            # =================================================
            # the bbox could be None
            smrc.utils.save_bbox_to_file(ann_path, [bbox])
            # =================================================

            tmp_img = cv2.imread(image_path)
            annotation_tool.draw_bboxes_from_file(tmp_img, ann_path)
            new_image_name = ann_path.replace('txt', 'jpg')
            print(f'ann_path = {ann_path}, new_image_name = {new_image_name}')
            cv2.imwrite(new_image_name, tmp_img)

        # sys.exit(0)


    # def save_fitted_bbox_for_checking(self, original_bbox_data, fitted_data, result_root_dir='fitting'):
    #     """
    #     transfer bbox to bbox_list
    #     :param original_bbox_data: data format [ [image_id, bbox], ... ]
    #     :param fitted_data: the fitted bbox data, data format [
    #             [image_id, bbox]
    #         ]
    #     :return: transfer and visualize all the original data and the fitted data
    #     for checking purpose, data format [
    #             [image_id, bbox_list]
    #         ]
    #     """
    #
    #     image_path = self.IMAGE_PATH_LIST[0]
    #     image_dir, video_dir, _ = smrc.line.split_image_path(image_path)
    #     smrc.line.generate_dir_if_not_exist(
    #         os.path.join(result_root_dir, video_dir)
    #     )
    #
    #     bbox_list_all_none_empty = [[image_id, bbox] for image_id, bbox in fitted_data if bbox is not None]
    #     # add the original data
    #     bbox_list_all_none_empty.extend(original_bbox_data)
    #
    #     # sort the data based on the first column (image_id)
    #     bbox_list_all_none_empty.sort(key=lambda r: r[0])
    #
    #     # print(f'bbox_list_all_none_empty = {bbox_list_all_none_empty}')
    #     annotation_tool = AnnotationTool()
    #     annotation_tool.CLASS_LIST = ['original', 'fitted']
    #     annotation_tool.CLASS_BGR_COLORS = [
    #         np.array(annotation_tool.GREEN),  # transfer tuple to array
    #         np.array(annotation_tool.RED)
    #     ]
    #
    #     for image_id, bbox in bbox_list_all_none_empty:
    #         # bbox_list = [bbox]
    #         image_path = self.IMAGE_PATH_LIST[image_id]
    #         print(image_id, bbox)
    #
    #         # generate annotation file name from image name
    #         ann_path = smrc.line.get_image_or_annotation_path(
    #             image_path, image_dir, result_root_dir, '.txt'
    #         )
    #         # =================================================
    #         # the bbox could be None
    #         smrc.line.save_bbox_to_file(ann_path, [bbox])
    #         # =================================================
    #
    #         tmp_img = cv2.imread(image_path)
    #         annotation_tool.draw_bboxes_from_file(tmp_img, ann_path)
    #         new_image_name = ann_path.replace('txt', 'jpg')
    #         print(f'ann_path = {ann_path}, new_image_name = {new_image_name}')
    #         cv2.imwrite(new_image_name, tmp_img)


    #
    # def estimate_bbox_center_trajectory(self, X_fitted, Y_fitted):
    #     """
    #     # estimate the maximum region of the all the center points of the fitted bbox
    #     :param X_fitted:
    #     :param Y_fitted:
    #     :return:
    #     """
    #     X_fitted, Y_fitted = [np.int0(X_fitted), np.int0(Y_fitted)]
    #
    #     x1, x2 = max(min(X_fitted), 0), min(max(X_fitted), self.active_image_width)
    #     y1, y2 = max(min(Y_fitted), 0), min(max(Y_fitted), self.active_image_width)
    #
    #     if x1 == x2:
    #         x2 = x1 + 2
    #
    #     if y1 == y2:
    #         y2 = y1 + 2
    #     bbox_center_rect = [x1, y1, x2, y2]
    #
    #     print(f'X_fitted={X_fitted}')
    #     print(f'Y_fitted={Y_fitted}')
    #     print(f'bbox_center_rect={bbox_center_rect}')
    #     return bbox_center_rect
    #
    #













