function ndi2c3d(fileIn, fileOut)

% Convert NDI Optotrak files to C3D files.
% BTK library needed.
%
% fileIn: path for input NDI file.
% fileOut: path for output C3D file.

% Read data from NDI
[pathstr,name,ext] = fileparts(fileIn);
[data,FPS,CollDate,CollTime] = ReadNdf(pathstr,[filesep,name,ext]);
% Get parameters
Nm = size(data,2);
Nf = size(data,3);
% Create new C3D
acq = btkNewAcquisition(Nm, Nf);
% Set acquisition frequency
btkSetFrequency(acq, FPS);
% Set points data
for i = 1 : btkGetPointNumber(acq)
    btkSetPointLabel(acq, i, ['M',num2str(i)]);
    btkSetPoint(acq, i, squeeze(data(:,i,:))');
end
% Write data to file
btkWriteAcquisition(acq, fileOut);
btkDeleteAcquisition(acq);

