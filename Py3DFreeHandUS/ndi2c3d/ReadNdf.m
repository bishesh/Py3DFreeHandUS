function [data,Sfrequency,CollDate,CollTime] = ReadNdf(datapath,datafile);
% READNDF (BodyMech.m):reads NDI-optotrak data files (*.NDF) from disk
%
% INPUT:
% datafile, *.NDF format (Northern-digital Data File)
%                    (see OPTOTRAK documentation for more information)
%       including extention
% datapath: computers' pathname (ending on a \) 
% PROCESS
% reads file
% strips header information
% reads measurementdata
% recodes missing markers to NaN
% OUTPUT
% data:  3D-matrix of measurement data (coordinates, markers, time) 
%        maesures in MILLIMETERS
% Sfrequency : sample frequency [1/sec]
% CollDate   : Collection date [dd/mm/yy]
% CollTime   : Collection time [hh:mm:ss]

% AUTHOR(S) AND VERSION-HISTORY
% Ver. 1.0 Creation (Tom Welter FBW-VU 1998)
% Ver. 1.1 bug removed: length of "usercomment" and "rest" (Jaap Harlaar AZVU 1998)
% Ver. 1.1.1 layout source code adjusted (Jaap Harlaar AZVU 1999)
% Ver. 1.1.2 reshape command introduced (Jaap Harlaar AZVU 1999)
% Ver. 1.2   name and input format changed (Jaap Harlaar AZVU 2000)
% Ver. 1.21 outputs collection date & -time(Jaap Harlaar VUmc 2002)


[fid,message]=fopen([datapath,datafile]);					% open file
if fid == -1
	error(message);
   return
end


% fid=fopen(name,'r');

filetype	=fread(fid,1,'char');		
items		=fread(fid,1,'int16'); 		% number of markers
subitems	=fread(fid,1,'int16'); 		% number of dimensions (normally = 3: x,y z )
numframes	=fread(fid,1,'int32');	% number of frames (i.e, sample instances) 
Sfrequency	=fread(fid,1,'float32');% sample frequency
usercomment	=char(fread(fid,60,'char'))'; 
sys_comment =char(fread(fid,60,'char'))'; % not used
descrp_file =char(fread(fid,30,'char'))'; % not used
cuttoff_frq =fread(fid,1,'int16'); % not used
CollTime    =char(fread(fid,10,'char'))'; 
CollDate    =char(fread(fid,10,'char'))'; 
rest		=fread(fid,71,'char'); 	% padding, not used

totalnum =items*subitems*numframes;


data=fread(fid,totalnum,'float32');  			% read all data


% I=find(data == -3.697314030288567e+028);	% NDF code for missing markers
I=find(data<=-10e20); % heuristic approach 

if ~isempty (I),
data(I)=NaN; % convert to NaN (Not_a_Number)
end

data=reshape(data,subitems,items,numframes);

fclose(fid);

return
% ============================================ 
% END ### READNDF ###

