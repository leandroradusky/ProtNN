'''
Created on Aug 12, 2020

@author: lradusky
'''

import pandas as pd
import numpy as np
import argparse

from biopandas.pdb import PandasPdb
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation as R
from math import *
from numpy.lib.scimath import arcsin, arccos

import sys
sys.path.append('../')

from Handlers.FileHandler import FileHandler
from Handlers.URLRetrieveHandler import URLRetrieveHandler

pd.set_option('mode.chained_assignment', None)

PDB_PATH = 'pdb/divided/pdb/'

ThreeOne = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F'
,'GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M'
,'ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T'
,'VAL':'V','TRP':'W','TYR':'Y'}

xAxis = [1,0,0]
yAxis = [0,1,0]
zAxis = [0,0,1]

def angl(v1, v2):
    a=v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]
    b=sqrt(pow(v1[0],2)+pow(v1[1],2)+pow(v1[2],2))
    c=sqrt(pow(v2[0],2)+pow(v2[1],2)+pow(v2[2],2))
    cos=a/(b*c)
    return arccos(cos)

aroundXMatrix = lambda angle: np.array([[1,0,0],
                                        [0, cos(angle),-sin(angle)],
                                        [0, sin(angle), cos(angle)]])

aroundYMatrix = lambda angle: np.array([[cos(angle),0,sin(angle)],
                                        [0,1,0],
                                        [-sin(angle),0,cos(angle)]])

aroundZMatrix = lambda angle: np.array([[cos(angle),-sin(angle),0],
                                        [sin(angle),cos(angle), 0],
                                        [0,0,1]])

def rotateVector(v,m):
    xRot=m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2];
    yRot=m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2];
    zRot=m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2];
    v[0]=xRot;
    v[1]=yRot;
    v[2]=zRot;
    return v

def rotate(angle, axis, coords):
    
    if(axis[0]==1 and axis[1]==0 and axis[2]==0):
        r=aroundXMatrix(angle)
    elif (axis[0]==0 and axis[1]==1 and axis[2]==0):
        r=aroundYMatrix(angle)
    elif (axis[0]==0 and axis[1]==0 and axis[2]==1):
        r=aroundZMatrix(angle)
    return rotateVector(coords,r)

def rotateDfRecord(df, index, xzProy, xyProy):
    x,y,z = df.loc[index,'x_coord'], df.loc[index,'y_coord'], df.loc[index,'z_coord']
    
    [x,y,z] = rotate(abs(angl(xAxis,xzProy)), yAxis, [x,y,z])
    [x,y,z] = rotate(abs(angl(xAxis,xyProy)), zAxis, [x,y,z])
    
    df.loc[index,'x_coord'] = round(x,1)
    df.loc[index,'y_coord'] = round(y,1)
    df.loc[index,'z_coord'] = round(z,1)

class PDBStructure( object ):
    
    def __init__(self, id, replaceExistent=False, workingFolder = "/data/"):
        '''
        Downloads the pdb and instantiate the atoms dataframe
        '''
        self.id = id
        
        filePath = workingFolder+PDB_PATH+self.id[1:3]+"/"+id+".pdb"
        if replaceExistent or not FileHandler.fileExists(filePath):
            # Retrieve file from internet
            fileLines = URLRetrieveHandler.RetrieveFileLines('http://www.rcsb.org/pdb/files/'+self.id+'.pdb')
            FileHandler.writeLines(filePath, fileLines)
        else:
            self.fileLines = FileHandler.getLines(filePath)
        
        ppdb = PandasPdb()
        ppdb_obj = ppdb.read_pdb(filePath)
        
        self.atoms = ppdb_obj.df['ATOM']
        self.hetatms = ppdb_obj.df['HETATM']
        
        
    def getResidueMesh(self, chain, res, radius=10., include_non_standard=False):
        '''
        Creates a mesh with the atoms in the surrounding
        @return: a dictionary with the non-empty points of the mesh, None if nonstandard not allowed and present
        '''
        
        # CA atom of the parameter residue is the center of the sphere
        # N-CA-C plane is the coordinates reference of the mesh
        
        CA = self.atoms.loc[ ( self.atoms['atom_name'] == 'CA' ) & \
                             ( self.atoms['residue_number'] == res ) & \
                             ( self.atoms['chain_id'] == chain ) ]
        C = self.atoms.loc[ ( self.atoms['atom_name'] == 'C' ) & \
                             ( self.atoms['residue_number'] == res ) & \
                             ( self.atoms['chain_id'] == chain ) ]
        N = self.atoms.loc[ ( self.atoms['atom_name'] == 'N' ) & \
                             ( self.atoms['residue_number'] == res ) & \
                             ( self.atoms['chain_id'] == chain ) ]
        
        # Selection waters, the NN features
        distances = squareform(pdist(self.hetatms.append(CA)[['x_coord', 'y_coord', 'z_coord']]))
        closeHetatoms = (distances[ -1 ] < radius) & (distances[ -1 ] > 3)
        try:
            closeHetatoms = self.hetatms.iloc[ closeHetatoms.T ]
        except:
            try:
                closeHetatoms = self.hetatms.iloc[ np.delete(closeHetatoms, -1, 0).T ]
            except:
                return None, None
            
        # If prohibited atoms present, return None
        if not include_non_standard and len(closeHetatoms.residue_name.unique()) > 1:
            return None, None

        # Selection of atoms, the NN features
        distances = squareform(pdist(self.atoms[['x_coord', 'y_coord', 'z_coord']]))
        closeAtoms = (distances[ CA.index ] < radius) & (distances[ CA.index ] > 3)
        closeAtoms = self.atoms.iloc[ closeAtoms.T ]
        
        # For further transform atoms to N-CA-C plane
        cax = CA.iloc[0].x_coord
        cay = CA.iloc[0].y_coord
        caz = CA.iloc[0].z_coord
        nx = N.iloc[0].x_coord - cax
        ny = N.iloc[0].y_coord - cay
        nz = N.iloc[0].z_coord - caz
        cx = C.iloc[0].x_coord - cax
        cy = C.iloc[0].y_coord - cay
        cz = C.iloc[0].z_coord - caz
        
        # Reference all atoms to origin
        closeAtoms.x_coord -= cax
        closeAtoms.y_coord -= cay
        closeAtoms.z_coord -= caz
        closeHetatoms.x_coord -= cax
        closeHetatoms.y_coord -= cay
        closeHetatoms.z_coord -= caz
        
        # Proyection of N to axis
        xyProy = [nx,ny,0]
        xzProy = [nx,0,nz]
        for i, cla in closeAtoms.iterrows():
            rotateDfRecord(closeAtoms, i, xzProy, xyProy)
        for i, cla in closeHetatoms.iterrows():
            rotateDfRecord(closeHetatoms, i, xzProy, xyProy)
        
        # Proyection of C to axis
        [cx,cy,cz] = rotate(abs(angl(xAxis,xzProy)), yAxis, [cx,cy,cz])
        [cx,cy,cz] = rotate(abs(angl(xAxis,xyProy)), zAxis, [cx,cy,cz])
        xyProy = [cx,cy,0]
        xzProy = [cx,0,cz]
        for i, cla in closeAtoms.iterrows():
            rotateDfRecord(closeAtoms, i, xzProy, xyProy)
        for i, cla in closeHetatoms.iterrows():
            rotateDfRecord(closeHetatoms, i, xzProy, xyProy)
        
        return closeAtoms, closeHetatoms
    
def main(inputPdb, workingFolder = "/data/"):
    pdbo = PDBStructure(inputPdb,replaceExistent=True, workingFolder=workingFolder)
    
    for chain in pdbo.atoms.chain_id.unique():
        # Features and water outputs
        df_atoms = pd.DataFrame()
        df_waters = pd.DataFrame()
        
        for res in pdbo.atoms.loc[pdbo.atoms.chain_id == chain].residue_number.unique():
            print( inputPdb, chain, res )
            
            try:
                mesh_atoms, mesh_waters=  pdbo.getResidueMesh( chain, res ) 
            
                if mesh_atoms is None: continue
                
                for i, mesh_atom in mesh_atoms.iterrows():
                    d = dict()
                    d["record"] = inputPdb+"_"+chain+"_"+str(res)
                    d["%s_%s_%s" % (mesh_atom.x_coord,mesh_atom.y_coord,mesh_atom.z_coord)] = mesh_atom.residue_name+"_"+mesh_atom.atom_name
                    df_atoms = df_atoms.append(d, ignore_index=True)
                
                for i, mesh_atom in mesh_waters.iterrows():
                    d = dict()
                    d["record"] = inputPdb+"_"+chain+"_"+str(res)
                    d["%s_%s_%s" % (mesh_atom.x_coord,mesh_atom.y_coord,mesh_atom.z_coord)] = 1
                    df_waters = df_waters.append(d, ignore_index=True)
                
                # To print
                #ppdb = PandasPdb()
                #ppdb.df['ATOM'] = mesh_atoms
                #ppdb.df['HETATM'] = mesh_waters
                #ppdb.to_pdb(path='/home/lradusky/Downloads/2ci2%s%s.pdb' % (chain, res))
            
            except:
                continue
        
        df_atoms.to_csv(workingFolder+"NN_digestion/%s_%s_atoms.df" % (inputPdb,chain), sep="\t")
        df_waters.to_csv(workingFolder+"NN_digestion/%s_%s_waters.df" % (inputPdb,chain), sep="\t")

    
if __name__ == '__main__':
    print( "Started" )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--pdb", type=str,
                        help="Pdb Code to be processed")
    parser.add_argument("-f","--folder", type=str,
                        help="Working folder to store data")
    args = parser.parse_args()
    
    main(args.pdb, args.folder)
    
    print( "Done" )
    