# pytest framework ---------------------------------------------
import requests
from rs4 import siesta
import time
import sys
import os
import xmlrpc.client 
from io import IOBase
import json
from . import apidoc

class Target:
    # f = Target ('http://localhost:')
    # f.get ("/v1/accounts/me")
    # == f.http.get ("/v1/accounts/me")
    # f.axios.get ("/v1/accounts/me")    
    # f.siesta.v1.accounts ("me").get ()

    def __init__ (self, endpoint, api_call = False):
        self.endpoint = endpoint        
        self.s = requests.Session ()
        self._api_call = api_call
        self.siesta = siesta.API (endpoint, reraise_http_error = False)
        if not self._api_call:
            self.axios = Target (endpoint, True)
        else:
            self.set_default_header ('Accept', "application/json")        
            self.set_default_header ('Content-Type', "application/json")
        
    @property
    def http (self):
        return self

    def set_default_header (self, k, v):
        self.s.headers.update ({k: v})

    def api (self, point = None):
        if point:
            return siesta.API (point, reraise_http_error = False)
        return self.siesta

    def __enter__ (self):
        return self
        
    def __exit__ (self, type, value, tb):
        self._close ()
        
    def __del__ (self):
        self._close ()
            
    def _close (self):
        pass

    def resolve (self, url):
        if url.startswith ("http://") or url.startswith ("https://"):
            return url
        else:
            return self.endpoint + url 

    def _request (self, method, url, *args, **kargs):
        url = self.resolve (url)
        rf = getattr (self.s, method)
        if args:
            args = list (args)
            request_data = args.pop (0)
            args = tuple (args)
        else:
            try:
                request_data = kargs.pop ('data')
            except KeyError:
                request_data = None    

        if isinstance (request_data, dict) and self._api_call:
            request_data = json.dumps (request_data)
        
        if request_data:   
            resp = rf (url, request_data, *args, **kargs)
        else:
            resp = rf (url, *args, **kargs)

        if resp.headers.get ('content-type', '').startswith ('application/json'):            
            try:
                resp.data = resp.json ()
            except:
                resp.data = {}    
            if "__spec__" in resp.data:
                reqh = kargs.get ('headers', {})
                reqh.update (self.s.headers)                
                apidoc.log_spec (method.upper (), url, resp.status_code, resp.reason, reqh, request_data, resp.headers, resp.data)            
        else:
            resp.data = resp.content    
        return resp

    def get (self, url, *args, **karg):
        return self._request ('get', url, *args, **karg)
        
    def post (self, url, *args, **karg):
        return self._request ('post', url, *args, **karg)
    
    def upload (self, url, data, **karg):
        files = {}
        for k in list (data.keys ()):
            if isinstance (data [k], IOBase):
                files [k] = data.pop (k)        
        return self._request ('post', url, files = files, data = data, **karg)

    def put (self, url, *args, **karg):
        return self._request ('put', url, *args, **karg)
    
    def patch (self, url, *args, **karg):
        return self._request ('patch', url, *args, **karg)
    
    def delete (self, url, *args, **karg):
        return self._request ('delete', url, *args, **karg)
    
    def head (self, url, *args, **karg):
        return self._request ('head', url, *args, **karg)
                
    def options (self, url, *args, **karg):
        return self._request ('options', url, *args, **karg)
    
    def rpc (self, url, proxy_class = None):
        return (proxy_class or xmlrpc.client.ServerProxy) (self.resolve (url))
    xmlrpc = rpc
    
    def jsonrpc (self, url, proxy_class = None):
        import jsonrpclib
        return (proxy_class or jsonrpclib.ServerProxy) (self.resolve (url))
    
    def grpc (self, url, proxy_class = None):
        raise NotImplementedError

if __name__ == "__main__":
    if "--gendoc" in sys.argv:
        apidoc.build_doc ()
    elif "--logon" in sys.argv:
        apidoc.truncate_log_dir ()
    elif "--logoff" in sys.argv:
        apidoc.truncate_log_dir (remove_only = True)    
    

