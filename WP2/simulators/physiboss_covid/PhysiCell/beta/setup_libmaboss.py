# This script attempts to download the libRoadrunner (binary) libraries and
# headers for your particular operating system. It puts them in a directory
# of your choosing (default = your home dir) and sets an environment variable
# to that location which can then be used by a PhysiCell Makefile.
#
# Author: Randy Heiland

import platform
import urllib.request
import os
import sys
import tarfile
import zipfile

if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), "addons", "PhysiBoSS", "MaBoSS-env-2.0")):
    print('libMaBoSS already installed')

else:
    os_type = platform.system()
    print('operating system = ',os_type)

    # Assume Windows
    mb_file = ""
    url = ""

    if os_type.lower() == 'darwin':
        mb_file = "libMaBoSS-osx64.tar.gz"
    elif os_type.lower().startswith("win") or os_type.lower().startswith("msys_nt") or os_type.lower().startswith("mingw64_nt"):
        mb_file = "libMaBoSS-win64.tar.gz"
    elif os_type.lower().startswith("linux"):
        mb_file = "libMaBoSS-linux64.tar.gz"
    else:
        print("Your operating system seems to be unsupported. Please submit a ticket at https://sourceforge.net/p/physicell/tickets/ ")
        sys.exit(1)

    url = "https://github.com/sysbio-curie/MaBoSS-env-2.0/releases/download/v2.5.2/" + mb_file

    fname = mb_file

    home = os.path.expanduser("~")
    print('libMaBoSS will now be installed into the addon PhysiBoSS addon folder:')
    dir_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'addons', 'PhysiBoSS')
    print(dir_name + '\n')
    # print('   - Press ENTER to confirm the installation')
    # print('   - Press CTL-C to abort the installation')

    if not os.path.exists(dir_name):
                try:
                    os.makedirs(dir_name)
                except:
                    print('Error trying to create directory: ',dir_name)
                    exit(1)
    print('Beginning download of libMaBoSS into ' + dir_name + ' ...')
    print(url)

    my_file = os.path.join(dir_name, fname)
    print('my_file = ',my_file)

    # if os_type.lower().startswith("win"):
    #     rrlib_dir = my_file[:-4]
    # else:  # darwin or linux
    #     rrlib_dir = my_file[:-7]
    # print('rrlib_dir = ',rrlib_dir)

    def download_cb(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = "\r%5.1f%% %*d / %d" % (
                percent, len(str(totalsize)), readsofar, totalsize)
            sys.stderr.write(s)
            if readsofar >= totalsize: # near the end
                sys.stderr.write("\n")
        else: # total size is unknown
            sys.stderr.write("read %d\n" % (readsofar,))

    urllib.request.urlretrieve(url, my_file, download_cb)

    os.chdir(dir_name)
    print('installing (uncompressing) the file...')
    
    try:
        tar = tarfile.open(mb_file)
        tar.extractall()
        tar.close()
        os.remove(mb_file)
    except:
        print('error untarring the file')
        exit(1)

    print('Done.\n')

    # # LIBRR_DIR := /Users/heiland/libroadrunner/roadrunner-osx-10.9-cp36m
    # print("Replace the following variables in your PhysiCell Makefile with these:\n")
    # #print("LIBRR_DIR := /Users/heiland/libroadrunner/roadrunner-osx-10.9-cp36m")
    # print("LIBRR_DIR := " + rrlib_dir)
    # if os_type == 'Windows':
    #     print("LIBRR_LIBS := " + rrlib_dir + "/bin\n")
    # else:
    #     print("LIBRR_LIBS := " + rrlib_dir + "/lib\n")
