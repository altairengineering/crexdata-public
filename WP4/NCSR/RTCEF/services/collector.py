import services.service as service
from configuration import get_service_parameters
import json
import os


class Collector(service.Service):
    # default values that should not be modified
    # change to lowercase because it is changed during conf parsing
    type = "collector"
    input_stream = "inputstream"
    dataset_versions = "datasetversions"
    engine_positions = "engineposition"
    assembly_reports = "assembleddatasets"
    score_reports = "modelscores"

    bucket_type = None
    buckets_number = None
    bucket_size = None
    collection_method = None
    k = None
    storage_path = None
    naming = None
    last_bucket = 0
    current_bucket = []
    stored_k = 0
    dataset_version = 0
    current_bucket_records = 0
    current_bucket_ts = 0
    cleaning_queue = []
    sync_offset = -1
    bucket_threshold = -1
    buckets_to_remove = []
    buckets_scores = []
    scores_times = []

    def __init__(self, configuration_file):
        super().__init__(configuration_file)
        service_parameters = get_service_parameters(self.config, self.type)

        self.bucket_type = service_parameters["buckettype"]
        self.buckets_number = int(service_parameters["bucketsnumber"])
        self.bucket_size = int(service_parameters["bucketsize"])
        self.collection_method = service_parameters["collectionmethod"]
        if self.collection_method == "lastk" or self.collection_method == "scorebased":
            self.k = int(service_parameters["k"])
        self.storage_path = service_parameters["storagepath"]
        self.naming = service_parameters["naming"]
        self.time_field = service_parameters["timefield"]

    def run(self):
        """
        During a run loop, the collector consumes a record
        from the input stream and according to the collection
        methodology it inserts it in the appropriate bucket.
        """

        # sync with engine
        rows = self.consume(self.engine_positions, max_records=500)
        if rows:
            for record in rows:
                self.sync_offset = record["offset"]

        # sync with factory dataset assembly
        rows = self.consume(self.assembly_reports, max_records=500)
        if rows:
            for record in rows:
                if record["buckets_range"][0] > self.bucket_threshold:
                    self.bucket_threshold = record["buckets_range"][0]

        # collection
        rows = self.consume(self.input_stream, max_records=500, max_offset=self.sync_offset)
        if rows:
            for record in rows:
                info = self.handle_record(record)
                if info is not None:
                    # print(info, flush=True)
                    self.buckets_scores.append([info, None])
                    # print(self.buckets_scores)
                    if self.stored_k > self.buckets_number:
                        bucket_to_del = self.last_bucket - self.buckets_number - 1
                        bucket_to_del_path = self.storage_path + self.naming + str(bucket_to_del)
                        self.add_to_deletion_queue((bucket_to_del_path, bucket_to_del))
                        self.stored_k = self.buckets_number

                    if self.collection_method == "lastk":
                        buckets_range = list(range(max(0, info['bucket_id'] - self.k + 1), info['bucket_id'] + 1))
                        # !!!dataset version message format!!!

                    if self.collection_method == "scorebased":
                        last_buckets_list = list(range(max(0, info['bucket_id'] - int(self.k / 2) + 1),
                                                       info['bucket_id'] + 1))
                        bucket_id_scores = []
                        range_start = info['bucket_id'] - (self.stored_k - 1)
                        range_end = info['bucket_id'] - int(self.k / 2) + 1
                        print((range_start, range_end), flush=True)
                        for i in range(range_start, range_end):
                            if self.buckets_scores[i][1] is not None:
                                bucket_id_scores.append((self.buckets_scores[i][0]["bucket_id"],
                                                         self.buckets_scores[i][1]["mcc"]))
                            else:
                                bucket_id_scores.append((self.buckets_scores[i][0]["bucket_id"], 0))

                        score_sorted_buckets = sorted(bucket_id_scores, key=lambda x: (-x[1], -x[0]))

                        first_buckets_list = [x[0] for x in score_sorted_buckets[0:int(self.k / 2)]]
                        buckets_range = sorted(first_buckets_list + last_buckets_list)
                        print(score_sorted_buckets, flush=True)

                    data_version_record = {
                        "path_prefix": self.storage_path + self.naming,
                        "buckets_range": buckets_range,
                        "version": self.dataset_version
                    }
                    self.dataset_version = self.dataset_version + 1
                    self.produce(self.dataset_versions, [data_version_record])
        self.delete_buckets()

        # read scores and assign them to buckets
        rows = self.consume(self.score_reports)
        if rows:
            for record in rows:
                scores = json.loads(record["batch_metrics"])
                score_time = record["timestamp"]
                self.scores_times.append([score_time, scores])
            # print(self.scores_times)
        for score_time in self.scores_times:
            time = score_time[0]
            scores = score_time[1]
            for i in range(len(self.buckets_scores)):
                if self.buckets_scores[i][0]["ts"] <= time <= self.buckets_scores[i][0]["te"]:
                    self.buckets_scores[i][1] = scores

    def handle_record(self, record):
        """
        The method accepts a record and inserts in the current bucket. In case the
        current bucket is full, the method saves the current bucket in the
        path = self.storage_path+self.naming+self.last_bucket, and creates a new empty
        bucket. In the case that a bucket is saved, the method also returns a dict with
        bucket information.

        :param record: a record (row) from the input stream
        :return: in case the bucket is full, the method return a
        dict with the characteristics of the bucket:
            path: path to the saved dataset file
            bucket_id: id of the bucket
            ts: first record timestamp
            te: last record timestamp
            len: number of records in bucket
        """
        ts = None
        te = None
        bucket_len = None
        old_bucket_path = None
        path = self.storage_path + self.naming + str(self.last_bucket)
        if self.bucket_type == "time":
            if self.current_bucket_records == 0:
                with open(path, "a") as dtout:
                    dtout.write(json.dumps(record) + "\n")
                    self.current_bucket_ts = int(record[self.time_field])
                    self.current_bucket_records = self.current_bucket_records + 1
            elif self.current_bucket_records > 0:
                bucket_len = self.current_bucket_records
                ts = self.current_bucket_ts
                tc = int(record[self.time_field])
                if tc - ts > self.bucket_size:
                    old_bucket_path = path
                    self.last_bucket = self.last_bucket + 1
                    self.stored_k = self.stored_k + 1
                    path = self.storage_path + self.naming + str(self.last_bucket)
                    # new bucket with new record
                    with open(path, "a") as dtout:
                        dtout.write(json.dumps(record) + "\n")
                        self.current_bucket_ts = int(record[self.time_field])
                        self.current_bucket_records = self.current_bucket_records + 1
                else:
                    # append in current bucket
                    with open(path, "a") as dtout:
                        self.current_bucket.append(record)
                        dtout.write(json.dumps(record) + "\n")
                        self.current_bucket_records = self.current_bucket_records + 1
            if old_bucket_path is not None:
                return {"path": old_bucket_path, "bucket_id": self.last_bucket - 1, "ts": ts, "te": tc - 1,
                        "len": bucket_len}
            else:
                return None
        elif self.bucket_type == "count":
            raise Exception("Count not supported yet")
            pass

    def cleanup(self):
        """
        Cleans all buckets.
        """
        min_bucket = max(0, self.stored_k - self.k)
        for bucket_to_del in range(min_bucket, self.stored_k + 1):
            bucket_to_del_path = self.storage_path + self.naming + str(bucket_to_del)
            if os.path.exists(bucket_to_del_path):
                os.remove(bucket_to_del_path)
            else:
                self.logger.info(
                    f"{self.type} service warning: The file in {bucket_to_del_path} does not exist")
            pass

    def add_to_deletion_queue(self, bucket):
        """
        Adds bucket path to deletion queue
        :param bucket:  path of bucket
        """
        self.cleaning_queue.append(bucket)

    def delete_buckets(self):
        """
        Deletes all buckets that are in deletion queue but are not locked.
        """
        self.buckets_to_remove.clear()
        for bucket_path_id in self.cleaning_queue:
            bucket_path = bucket_path_id[0]
            bucket_id = bucket_path_id[1]
            if os.path.exists(bucket_path):
                if not os.path.exists(bucket_path + ".lock") and bucket_id < self.bucket_threshold:
                    os.remove(bucket_path)
                    self.buckets_to_remove.append(bucket_path_id)
            else:
                self.logger.info(
                    f"{self.type} service warning: The file in {bucket_path} does not exist")
        for bucket_path_id in self.buckets_to_remove:
            self.cleaning_queue.remove(bucket_path_id)
