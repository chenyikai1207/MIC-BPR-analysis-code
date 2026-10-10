"""Read selected MAT-v5 metadata with stdlib only; no signal or filtering code.

This deliberately rejects unsupported classes/types and MAT7.3. It is a narrow
independent audit reader, not a replacement EEGLAB processing implementation.
"""
import math, struct, zlib
from pathlib import Path

TYPE_FORMAT={1:'b',2:'B',3:'h',4:'H',5:'i',6:'I',7:'f',9:'d',12:'q',13:'Q'}

def element(buf,pos,endian):
    assert pos+8<=len(buf),'Incomplete MAT element tag'
    kind16,small_size=struct.unpack_from(endian+'HH',buf,pos)
    if kind16 and 0<small_size<=4:
        kind=kind16
        return kind,buf[pos+4:pos+4+small_size],pos+8
    kind,size=struct.unpack_from(endian+'II',buf,pos)
    assert pos+8+size<=len(buf),'Incomplete MAT element payload'
    # Compressed top-level elements are commonly not padded in MAT-v5 writers.
    nextpos=pos+8+size if kind==15 else pos+8+((size+7)//8)*8
    return kind,buf[pos+8:pos+8+size],nextpos

def numeric(kind,buf,endian):
    assert kind in TYPE_FORMAT,'Unsupported MAT numeric type '+str(kind)
    fmt=endian+TYPE_FORMAT[kind]
    assert len(buf)%struct.calcsize(fmt)==0
    return [v[0] for v in struct.iter_unpack(fmt,buf)]

def matrix(buf,endian,header_only=False):
    kind,flags,pos=element(buf,0,endian);assert kind==6 and len(flags)>=8
    cls=struct.unpack_from(endian+'I',flags)[0]&255
    kind,dimbytes,pos=element(buf,pos,endian);assert kind==5
    dims=numeric(kind,dimbytes,endian);n=math.prod(dims)
    kind,namebytes,pos=element(buf,pos,endian);assert kind in (1,2)
    name=namebytes.decode('utf8')
    if header_only:return name,cls,dims
    if cls==2:
        kind,sizebytes,pos=element(buf,pos,endian);assert kind==5
        fieldwidth=numeric(kind,sizebytes,endian)[0]
        kind,names,pos=element(buf,pos,endian);assert kind in (1,2) and fieldwidth>0
        assert len(names)%fieldwidth==0
        fields=[names[j:j+fieldwidth].split(b'\0',1)[0].decode('utf8')
                for j in range(0,len(names),fieldwidth)]
        values=[]
        for index in range(n):
            row={}
            for field in fields:
                kind,payload,pos=element(buf,pos,endian);assert kind==14
                # A zero-size miMATRIX is an explicitly empty struct field.
                row[field]=matrix(payload,endian)[1] if payload else []
            values.append(row)
        return name,values[0] if n==1 else values
    if cls==1:
        values=[]
        for index in range(n):
            kind,payload,pos=element(buf,pos,endian);assert kind==14
            values.append(matrix(payload,endian)[1])
        return name,values[0] if n==1 else values
    if cls==4:
        if n==0:return name,''
        kind,payload,pos=element(buf,pos,endian)
        if kind==16:value=payload.decode('utf8')
        elif kind in (4,17):value=payload.decode('utf-16-le' if endian=='<' else 'utf-16-be')
        elif kind in (1,2):value=payload.decode('latin1')
        elif kind==18:value=payload.decode('utf-32-le' if endian=='<' else 'utf-32-be')
        else:raise ValueError('Unsupported MAT character type '+str(kind))
        return name,value
    if 6<=cls<=15:
        if n==0:return name,[]
        kind,payload,pos=element(buf,pos,endian);values=numeric(kind,payload,endian)
        assert len(values)==n
        return name,values[0] if n==1 else values
    raise ValueError('Unsupported MAT class '+str(cls))

def read_selected(path,names):
    raw=Path(path).read_bytes();assert len(raw)>=128
    assert raw[:16].startswith(b'MATLAB 5.0'), 'Only MAT-v5 is supported'
    marker=raw[126:128];assert marker in (b'IM',b'MI')
    endian='<' if marker==b'IM' else '>'
    result={};pos=128
    while pos<len(raw):
        kind,payload,pos=element(raw,pos,endian)
        if kind==15:
            expanded=zlib.decompress(payload)
            kind,payload,last=element(expanded,0,endian)
            assert kind==14 and last==len(expanded)
        assert kind==14
        name,cls,dims=matrix(payload,endian,True)
        if name in names:
            assert name not in result
            result[name]=matrix(payload,endian)[1]
    return result,{'MAT_endian':endian,'bytes':len(raw),'selected_keys':sorted(result)}
