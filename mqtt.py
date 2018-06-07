import paho.mqtt.client as mqtt
from mqtt_utils import LoadLogger
import os
import sys
import pandas as pd
import datetime
from time import time as tm
from threading import Timer
from gdcb_explore import GDCBExplorer

'''
Modified:
  2018-05-22  Added repeating Timer Class
              Short idle message + buffer status report
              Added buffer time to live  
              Added stats display on buffer
  2018-05-23  Added reset idle message
              Added session upload counter
'''

__VERSION__ = "0.98.2"

class RepeatingTimer(object):

    def __init__(self, interval, f, *args, **kwargs):
        self.interval = interval
        self.f = f
        self.args = args
        self.kwargs = kwargs

        self.timer = None

    def callback(self):
        self.f(*self.args, **self.kwargs)
        self.start()

    def cancel(self):
        self.timer.cancel()

    def start(self):
        self.timer = Timer(self.interval, self.callback)
        self.timer.start()

class GMqttClient():

  def __init__(self, name = "GMqtt1", config_file = "config.txt", debug = False, debug_level = 1):
    """
    name - object name
    config_file - JSON config file
    debug - used for debug purpose in oder that client to not consume 
            all messages when run in test mode
    debug_level - verbosity level, if greater than 1 individual logging for 
                  each received message 
    """
    self.__version__ = __VERSION__
    self.init_cache_timestamp = 0
    self.init_gdcb_timestamp = 0
    self.name = name
    self.config_file = config_file
    self.logger = LoadLogger(lib_name=self.name,
                             config_file=self.config_file,
                             log_suffix="",
                             TF_KERAS=False,
                             HTML=True)
    self.config_data = self.logger.config_data
    self._init_gdcb_explorer()
    self._get_config_data()

    self.df_crt_batch = pd.DataFrame(columns = self.cols_list)
    self.file_ct = 0
    self.num_received_msg = 0
    self.DEBUG = debug
    self.DEBUG_COUNTER = 1

    self.debug_level = debug_level
    self.start_recv_minibatch = 0
    self.end_recv_minibatch = 0
    self.minibatch_ct = 0

    self.global_start_time = tm()
    self.flag_connected = False
    self.flag_in_log = False
    self.flag_display_buffstats = False

    self.last_msg_time = 0
    self.idle_time = tm() - self.last_msg_time
    self.session_upload_counter = 0
    
    self.log("Starting MQTTClient ver.{}".format(self.__version__))
    self.log("Debug level: {}".format(debug_level))

    self._init_cache()
    self._setup_status_timer()
    return


  def _setup_status_timer(self):
    self.alive_timer = RepeatingTimer(self.check_time * 60, 
      self._display_status_service_run)
    self.alive_timer.start()


  def _display_status_service_run(self):
    seconds_alive = tm() - self.global_start_time
    self.idle_time = tm() - self.last_msg_time
    if self.flag_connected and (self.idle_time > self.check_time * 60):
      self.log("AliveThread: topic {}, svr {}, {:.3f} hrs (idle {:.3f} min) [{}/{}]".
        format(self.topic_tree[:-2], self.server, seconds_alive / 3600, 
          self.idle_time / 60, self.df_crt_batch.shape[0], self.batch_size))
      if self.idle_time >= self.buffer_ttl * 60 and self.df_crt_batch.shape[0] > 0:
        self.log("Forcing dispatch for {} msgs".format(self.df_crt_batch.shape[0]))
        self._display_buffer_stats(self.df_crt_batch)
        self._dispatch_and_clean()
    return


  def _get_config_data(self):
    self.base_folder  = self.config_data['BASE_FOLDER']
    self.app_folder   = self.config_data['APP_FOLDER']
    self.server       = self.config_data['SERVER']
    self.port         = int(self.config_data['PORT'])
    self.keep_alive   = int(self.config_data['KEEP_ALIVE'])
    self.topic_token  = self.config_data['TOPIC_PIDS']
    self.topic_tree   = self.topic_token + "/#"
    self.batch_size   = int(self.config_data['BATCH_SIZE'])
    self.path_tokens  = self.config_data['PATH']
    self.dbg_tokens   = self.config_data['DEBUG_CODES']
    self.h_init_cache = self.config_data['HOURS_REINIT_CACHE']
    self.h_init_gdcb  = self.config_data['HOURS_REINIT_GDCB']
    self.cols_list    = self.path_tokens + [self.gdcb.raw_sval_field, self.gdcb.raw_time_field]

    self.check_time       = float(self.config_data['MINUTES_CHECK_INTERVAL'])
    self.num_log_msgs     = int(self.config_data['NR_LOG_MESSAGES']) 
    self.dlevel_print_all = int(self.config_data['DEGUG_LEVEL_PRINT_ALL'])
    self.buffer_ttl = int(self.config_data['BUFFER_TTL']) 
    
    self.valid_topics = [self.topic_token]
    return

  def _display_buffer_stats(self, df):

    srr = df[['VIN', 'Code']].groupby(['VIN', 'Code']).size()
    unique_VINs = df['VIN'].unique()
    self.log(" Display {} msgs for {} VINs".format(df.shape[0], 
      unique_VINs.shape[0]))
    for i, s in enumerate(srr):
      self.log("  VIN/Code: {}/{}: {} msgs".format(srr.index[i][0], 
        srr.index[i][1], s))
    return


  def _init_gdcb_explorer(self):
    t_sec = tm()
    if (self.init_gdcb_timestamp == 0) or ((t_sec - self.init_gdcb_timestamp) >= self.h_init_gdcb * 3600):
      self.gdcb = GDCBExplorer(self.logger, load_data = False)
      self.init_gdcb_timestamp = tm()
    return

  def _init_cache(self):
    t_sec = tm()
    if (self.init_cache_timestamp == 0) or ((t_sec - self.init_cache_timestamp) >= self.h_init_cache * 3600):
      str_query = 'SELECT B.CarID, A.VIN, B.Code, B.CodeID, C.Mult, C.[Add], C.Units  FROM ' +\
                  self.gdcb.config_data["CARS_TABLE"] + " A, " +\
                  self.gdcb.config_data["CARSXCODES_TABLE"] + " B, " +\
                  self.gdcb.config_data["PREDICTOR_TABLE"] + " C " +\
                  ' WHERE A.ID = B.CarID AND B.CodeID = C.ID'
  
      self.df_cache = self.gdcb.sql_eng.Select(str_query)
      self.logger.SaveDataframe(self.df_cache, fn = 'DATA_CACHE')
      self.init_cache_timestamp = tm()
    return
  
  def log(self, msg, show_time =  False, show = True):
    if show and not self.flag_in_log:
      self.flag_in_log = True
      self.logger.VerboseLog(msg, show_time = show_time)
      self.flag_in_log = False

  def _on_connect(self, client, userdata, flags, rc):

    self.log("Connected to server {} on port {} with keepalive {}".format(
      self.server, self.port, self.keep_alive))
    self.log("\t Client: {}".format(client))
    self.log("\t User Data: {}".format(userdata))
    self.log("\t Flags: {}".format(flags))
    self.log("\t Result code: {}".format(rc))

    client.subscribe(self.topic_tree)
    self.flag_connected = True

    return

  def _on_disconnect(self, client, userdata, rc):
    if rc != 0:
      self.log("Client disconnected unexpectedly. Disconnection code = {}.".format(rc))
    else:
      self.log("Client disconnected.")
    return

  def _on_message(self, client, userdata, msg):

    self.last_msg_time = tm()
    self.num_received_msg += 1
    if self.DEBUG and (self.num_received_msg >= self.DEBUG_COUNTER):
      self.client.disconnect()

    if msg.retain == 1:
      self.log("Message skipped due to retain=1")
      return

    if self.minibatch_ct == 0:
      self.start_recv_minibatch = tm()

    if self.minibatch_ct < self.num_log_msgs:
      self.minibatch_ct += 1
    else:
      self.end_recv_minibatch = tm()
      self.log("Received {} messages (last received topic {}) in {:.2f}s".
        format(self.minibatch_ct, msg.topic, self.end_recv_minibatch - self.start_recv_minibatch))
      self.minibatch_ct = 0
      self.flag_display_buffstats = True


    if self.debug_level >= self.dlevel_print_all:
      self.log("Received mesage: Topic=[{}]; Payload=[{}]; Tstmp=[{}]; Index=[{}]".format(
        msg.topic, msg.payload, msg.timestamp, self.num_received_msg))

    self.register_message(msg.topic, msg.payload)
    return

  def _on_subscribe(self, client, userdata, mid, granted_qos):

    self.log("Subscribe on {} accepted:".format(self.topic_tree))
    self.log("\t Client: {}".format(client))
    self.log("\t User Data: {}".format(userdata))
    self.log("\t Mid: {}".format(mid))
    self.log("\t Granted_qos: {}".format(granted_qos))
    
  def DispatchSavedBatch(self, file_path, dispatch = True):
    self.df_loaded_batch = pd.read_csv(file_path)
    self.log("Loaded saved batch [..{}].".format(file_path[-35:]))
    """
    if dispatch:
      self._dispatch(self.df_loaded_batch, save_to_disk=False)
    """
    return


  def _dispatch_and_clean(self):
    _success = self._dispatch(self.df_crt_batch)

    if _success:
      self.df_crt_batch = self.df_crt_batch[0:0]
      self.minibatch_ct = 0
      
    return


  def register_message(self, topic_data, payload_data):
    topic_values = [data for data in topic_data.split('/') if data != ""]
    self.log("{}".format(topic_values))
    if topic_values[0] in self.valid_topics:
      if self.idle_time > self.check_time * 60:
        pass
        #self.log("Message received. Idle time will reset.")
      msg_dict = {}
      for i in range(1,len(topic_values)):
        msg_dict[self.path_tokens[i-1]] = topic_values[i]
      msg_dict[self.cols_list[-2]] = payload_data.decode("utf-8")
      recv_tmstp = datetime.datetime.now()
      recv_tmstp = recv_tmstp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
      msg_dict[self.cols_list[-1]] = recv_tmstp
  
      self.df_crt_batch = self.df_crt_batch.append(pd.DataFrame(msg_dict, index=[0]), 
        ignore_index = True)

      if self.flag_display_buffstats:
        self._display_buffer_stats(self.df_crt_batch.tail(1000))
        self.flag_display_buffstats = False
      
      if self.df_crt_batch.shape[0] >= self.batch_size:
        self._display_buffer_stats(self.df_crt_batch)
        self._dispatch_and_clean()
    else:
        self.log("Received invalid topic {}!".format(topic_values[0]))
    return


  def _dispatch(self, df, save_to_disk=True):
  
    df_to_dispatch = df.copy()

    filename  = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    filename += "_" + str(self.batch_size) + "batch.csv"

    if os.path.isfile(filename):
      filename  = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
      filename += "_" + str(self.file_ct)
      filename += "_" + str(self.batch_size) + "batch.csv"
      self.file_ct += 1
    else:
      self.file_ct = 0       
    
    self._init_gdcb_explorer()    
    self._init_cache()

    _success = self.write_to_database(df_to_dispatch, save_to_disk)
    if _success:
      self.session_upload_counter += df_to_dispatch.shape[0]
      df_to_dispatch = df_to_dispatch[0:0]
      self.log("Cleaning cache. So far uploaded {} msgs this sess".
        format(self.session_upload_counter))
    else:
      self.log("Cache preserved!")

    return _success

  def write_to_database(self, df, save_to_disk):
    df_joined = pd.merge(df, self.df_cache, how='left', on=['VIN', 'Code'])
    df_joined.drop('VIN', axis=1, inplace=True)
    df_joined = df_joined.fillna(value={'CarID':-1})
    _success = self.gdcb.DumpDfToRawData(df_joined, save_to_disk = save_to_disk)
    return _success
    

  def setup_connection(self):

    self.client = mqtt.Client()
    self.client.on_connect    = self._on_connect
    self.client.on_message    = self._on_message
    self.client.on_subscribe  = self._on_subscribe
    self.client.on_disconnect = self._on_disconnect

    self.client.connect(self.server, self.port, self.keep_alive)

    self.client.loop_forever()

if __name__ == "__main__":

  if (len(sys.argv) < 2):
    vlevel = 1
  else:
    vlevel = int(sys.argv[1])

  vebosity_level = max(1, vlevel)
  gdc = GMqttClient(debug_level = vlevel)
  #gdc.DispatchSavedBatch('batches/29-01-2018_21-50-56_100batch.csv')
  #gdc.write_to_database(gdc.df_loaded_batch)
  gdc.setup_connection()
