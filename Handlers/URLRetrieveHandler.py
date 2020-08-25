'''
Created on Jul 12, 2013
@author: leandro
'''

import urllib
from Handlers.FileHandler import FileHandler
from Handlers.SystemHandler import SystemHandler

PROXY = ""

class URLRetrieveHandler(object):
    
    @staticmethod
    def RetrieveFileLines(url):
        URLRetrieveHandler.RetrieveFileToDisk(url, "/tmp/", "download")
        return FileHandler.getLines("/tmp/download")
        
    @staticmethod
    def RetrieveFileToDisk(url,path, filename=""):
        dir = FileHandler.ensureDir(path)
        if PROXY != "":
            command = "cd %s; wget %s -e use_proxy=yes -e http_proxy=%s -e ftp_proxy=%s %s 2> /dev/null" % \
            (dir,url,PROXY,PROXY, "" if filename == "" else " -O %s " % filename)
        else:
            command = 'cd %s; wget "%s" %s 2> /dev/null' % \
            (dir,url, "" if filename == "" else " -O  %s " % filename)
        SystemHandler.getCommandResult(command)