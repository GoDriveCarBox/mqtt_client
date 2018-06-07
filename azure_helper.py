# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 19:43:09 2017
"""

from __future__ import print_function
import pandas as pd
import pyodbc
import urllib
import json
from sqlalchemy import create_engine
import datetime
import time as tm
import os

__author__     = "Andrei Ionut DAMIAN"
__copyright__  = "Copyright 2007 4E Software"
__credits__    = ["Andrei Simion"]
__license__    = "GPL"
__version__    = "1.3.3"
__maintainer__ = "Andrei Ionut DAMIAN"
__email__      = "damian@4esoft.ro"
__status__     = "Production"
__library__    = "AZURE SQL HELPER"
__created__    = "2017-01-25"
__modified__   = "2017-06-01"
__lib__        = "SQLHLP"


def start_timer():
    return tm.time()

def end_timer(start_timer):
    return(tm.time()-start_timer)

def print_progress(str_text):
    print("\r"+str_text, end='\r', flush=True)
    return

class MSSQLHelper:
  def __init__(self,
               config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sql_config.txt'),
               parent_log = None):

      self.DEBUG = 1
      self.debug_str_size = 35


      self.parent_log = parent_log
      self.MODULE = '[{} v{}]'.format(__library__,__version__)
      self._logger("INIT "+self.MODULE)
      cfg_file = open(config_file)
      config_data = json.load(cfg_file)
      cfg_file.close()
      self.driver   = config_data["driver"]
      self.server   = config_data["server" ]
      self.database = config_data["database"]
      self.username = config_data["username"]
      self.password = config_data["password"]
      self.cwd = os.path.abspath(os.path.dirname(__file__))

      try:
          self.dfolder  = os.path.join(os.path.abspath(os.path.dirname(__file__)), config_data["datafolder"])
      except:
          self.dfolder = "save"
          self.dfolder = os.path.join(self.cwd,self.dfolder)

      self.data_folder = self.dfolder

      self.dfolder = os.path.join(self.dfolder,"db_cache")

      if not os.path.isdir(self.dfolder):
          self._logger("Creating data folder:{}".format(
                              self.dfolder[-self.debug_str_size:]))
          os.makedirs(self.dfolder)
      else:
          self._logger("Using data folder:...{}".format(
                  self.dfolder[-self.debug_str_size:]))

      self.connstr = 'DRIVER=' + self.driver
      self.connstr+= ';SERVER=' + self.server
      self.connstr+= ';DATABASE=' + self.database
      self.connstr+= ';UID=' + self.username
      self.connstr+= ';PWD=' + self.password
      self.engine = None


      sql_params = urllib.parse.quote_plus(self.connstr)

      try:
          self._logger("ODBC Conn: {}...".format(self.connstr[:self.debug_str_size]))
          self.conn = pyodbc.connect(self.connstr,
                                     timeout = 2)
          self.engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_params,
                                      connect_args={'connect_timeout': 2})
          self._logger("Connection created on "+self.server)
      except Exception as err: #pyodbc.Error as err:
          self._logger("FAILED ODBC Conn!")
          self.HandleError(err)
      return


  def Select(self,str_select, caching = False, convert_ascii = None):
      df = None
      try:
          str_fn = "".join(["_" if x in " ,;()*\\\\/[].><" else x for x in str_select])
          str_fn = str_fn.replace("__","_").replace("__","_")
          str_fn += ".csv"
          str_fn = os.path.join(self.dfolder,str_fn)
          if self.DEBUG>1:
              self._logger("Using datafile: {}".format(str_fn))
          t0 = tm.time()
          if (not os.path.isfile(str_fn)) or (not caching):
              fmt_sql = " ".join(str_select.split())[:80]
              if self.DEBUG>0:
                  self._logger("Downloading data [{}..] ...".format(fmt_sql[:30]))
              else:
                  self._logger("Downloading data ...")
              df = pd.read_sql(str_select, self.conn)
              if convert_ascii != None:
                  # now convert columns to ascii
                  for col in convert_ascii:
                      df[col] = df[col].apply(lambda x: ''.join(
                              [" " if ord(i) < 32 or ord(i) > 126 else i
                                   for i in x]))
              if caching:
                  if self.DEBUG>0:
                      self._logger("Saving to [..{}]...".format(str_fn[-self.debug_str_size:]))
                  else:
                      self._logger("Saving cache...")
                  df.to_csv(str_fn, index = False)
          else:
              if self.DEBUG>0:
                  self._logger("Loading file [..{}] ...".format(str_fn[-self.debug_str_size:]))
              else:
                  self._logger("Loading file ...")
              df = pd.read_csv(str_fn)
          nsize = self.GetSize(df) / float(1024*1024)
          t1 = tm.time()
          tsec = t1-t0
          tmin = float(tsec) / 60
          self._logger("Dataset loaded: {:.2f}MB in {:.1f}s({:.1f}m) {} rows".format(
                       nsize,
                       tsec,
                       tmin,
                       df.shape[0],
                       str_select))
          if self.DEBUG>1:
              self._logger("Dataset head(3):\n{}".format(df.head(2)))
          #self._logger("  READ TABLE time: {:.1f}s ({:.2f}min)".format(tsec,tmin))
      except Exception as err: #pyodbc.Error as err:
          self.HandleError(err)
      return df


  def ReadTable(self, str_table, caching=False):
    str_select = "SELECT * FROM ["+str_table+"]"
    return self.Select(str_select, caching)

  def GetEmptyTable(self, str_table):
    str_select = "SELECT TOP (1) * FROM ["+str_table+"]"
    return self.Select(str_select)[0:0]

  def CustomSelect(self, str_table, CarID, Code):
    str_select = ("SELECT * FROM [%s] WHERE CarID=%d AND Code='%s'") %\
      (str_table, CarID, Code)
    return self.Select(str_select).loc[0]

  def ExecInsert(self, sInsertQuery):
      try:
          t0 = tm.time()
          cursor = self.conn
          cursor.execute(sInsertQuery)
          self.conn.commit()
          t1 = tm.time()
          tsec = t1-t0
          tmin = float(tsec) / 60
          self._logger("EXEC SQL  time: {:.1f}s ({:.2f}min)".format(tsec,tmin))
      except Exception as err: #pyodbc.Error as err:
          self.HandleError(err)
      return


  def SaveTable(self, df, sTable, log_sql=False): ### see if we can save sql text...
    dfsize = self.GetSize(df) / (1024*1024)
    _success = True
    try:
        self._logger("SAVING TABLE [APPEND]({:,} records {:,.2f}MB)...".format(
                     df.shape[0],
                     dfsize))
        t0 = tm.time()
        df.to_sql(sTable,
                  self.engine,
                  index = False,
                  if_exists = 'append')
        t1 = tm.time()
        tsec = t1-t0
        tmin = float(tsec) / 60
        self._logger("DONE SAVE TABLE. Time = {:.1f}s ({:.2f}min)".format(tsec,tmin))
    except Exception as err: #pyodbc.Error as err:
        self.HandleError(err)
        _success = False

    return _success

  def OverwriteTable(self, df, sTable):
    dfsize = self.GetSize(df) / (1024*1024)
    try:
        self._logger("SAVING TABLE [OVERWRITE]({:,} records {:,.2f}MB)...".format(
                     df.shape[0],
                     dfsize))
        t0 = tm.time()
        df.to_sql(sTable,
                  self.engine,
                  index = False,
                  if_exists = 'replace')
        t1 = tm.time()
        tsec = t1-t0
        tmin = float(tsec) / 60
        self._logger("DONE SAVE TABLE. Time = {:.1f}s ({:.2f}min)".format(tsec,tmin))
    except Exception as err: #pyodbc.Error as err:
        self.HandleError(err)
    return

  def Close(self):
    self.conn.close()
    return


  def HandleError(self, err):
      strerr = "ERROR: "+ str(err) #[:50]
      self._logger(strerr)
      return

  def GetSize(self,df):
      dfsize = df.values.nbytes + df.index.nbytes + df.columns.nbytes
      return dfsize


  def _logger(self, logstr, show = True):

      if self.parent_log != None:
          logstr = "[{}] ".format(__lib__) + logstr
          self.parent_log._logger(logstr,show)
      else:
          if not hasattr(self, 'log'):
              self.log = list()
          nowtime = datetime.datetime.now()
          strnowtime = nowtime.strftime("[{}][%Y-%m-%d %H:%M:%S] ".format(__lib__))
          logstr = strnowtime + logstr
          self.log.append(logstr)
          if show:
              print(logstr, flush = True)
      return

  def ClearCache(self):
    self._logger("Cleaning DB cache ...")
    self.EmptyFolder(self.dfolder)
    self._logger("Done cleaning DB cache.")
    return

  def EmptyFolder(self, sFolder):
    for the_file in os.listdir(sFolder):
        file_path = os.path.join(sFolder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    return



  def __exit__(self, exc_type, exc_val, exc_tb):
      self.conn.close()
      self._logger("__exit__")
      return

if __name__ == '__main__':

    print("ERROR: MSSQLHelper is library only!")
