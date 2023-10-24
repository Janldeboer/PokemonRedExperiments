import hnswlib
import numpy as np


class KnnHandler:
    def __init__(self, num_elements=20000, vec_dim=4320):
        self.num_elements = num_elements
        self.vec_dim = vec_dim

        self.base_explore = 0
        self.sim_frame_dist = 2000000

        self.count = 0

        self.knn_index = self.create_index()

    def create_index(self):
        new_index = hnswlib.Index(
            space="l2", dim=self.vec_dim
        )  # possible options are l2, cosine or ip
        new_index.init_index(max_elements=self.num_elements, ef_construction=100, M=16)
        return new_index

    def is_frame_novel(self, vec):
        if self.count == 0:
            print("adding first frame to knn index")
            return True
        else:
            flat = vec.flatten().astype(np.float32)
            return self.knn_index.knn_query(flat, k=1)[1][0] > self.sim_frame_dist

    def update_frame_knn_index(self, frame_vec):
        if self.is_frame_novel(frame_vec):
            new_frame_id = self.count
            identifiers = np.array([new_frame_id])
            self.knn_index.add_items(
                frame_vec.flatten().astype(np.float32), identifiers
            )
            self.count += 1

    def number_of_frames(self):
        return self.knn_index.get_current_count()

    def correct_count(self):
        correct = self.knn_index.get_current_count()
        if correct != self.count:
            print(f"count is {self.count} but knn index says {correct}")
        self.count = correct

    """
    TODO: Check if this is still needed

    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        event_flags_start = 0xD747
        event_flags_end = 0xD886
        museum_ticket = (0xD754, 0)
        base_event_flags = 13
        return max(
            sum(
                [
                    self.bit_count(self.read_m(i))
                    for i in range(event_flags_start, event_flags_end)
                ]
            )
            - base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
        0,
    )
    """
