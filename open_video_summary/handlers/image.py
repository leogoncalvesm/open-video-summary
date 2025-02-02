from numpy import argsort, dot, transpose
from open_video_summary.entities.image import Keyframe


class KeyframeHandler:
    @staticmethod
    def num_matches(kf: Keyframe, other: Keyframe, threshold: float = 0.95) -> int:
        num_match = 0
        d1_t, d2_t = map(transpose, (kf.descriptor, other.descriptor))

        for i, desc in enumerate(kf.descriptor):
            sim = dot(desc, d2_t)
            self_match = argsort(-sim)[0]

            if sim[self_match] >= threshold:
                match_feature = other.descriptor[self_match]
                sim_check = dot(match_feature, d1_t)
                other_match = argsort(-sim_check)[0]

                num_match += (sim_check[other_match] >= threshold) and (
                    other_match == i
                )

        return num_match

    @staticmethod
    def is_keyframe(
        keyframe: Keyframe,
        keyframe_list: list[Keyframe],
        min_keypoints_diff_ratio: float = 0.6,
        min_descriptors_diff_ratio: float = 0.1,
    ) -> bool:
        if not keyframe_list:
            return True

        return sum(
            (
                abs(kf.descriptor_size - kf.descriptor_size)
                >= kf.descriptor_size * min_keypoints_diff_ratio
            )
            or (
                KeyframeHandler.num_matches(keyframe, kf)
                < (min_descriptors_diff_ratio * kf.descriptor_size)
            )
            for kf in keyframe_list
        ) == len(keyframe_list)
