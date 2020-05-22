clear all, close all, clc

% work directories
bag_directory = 'bag';
csv_directory = 'csv';

% find the total available bag files
path = [bag_directory,'\'];
files = dir([path,'\*.bag']);
len = length (files);

disp('Reading Bag files');

for idx = 1:len
    name = files(idx).name;
    filename = name(1:end-4);
    
    % read bag files
    bag = rosbag(['bag/',filename,'.bag']);
    
    %IMU data
    bSel_ori = select(bag,'Topic','/mavros/imu/data');
    imu_data = cell2mat(readMessages(bSel_ori,'DataFormat','struct'));
    time = cell2mat((extractfield(imu_data,'Header'))');
    stamps = cell2mat((extractfield(time,'Stamp'))');
    Orientation = cell2mat((extractfield(imu_data,'Orientation'))');
    Angular_velocity=cell2mat(extractfield(imu_data,'AngularVelocity')');
    Linear_acceleration=cell2mat(extractfield(imu_data,'LinearAcceleration'))';
    
    
    Seconds = double(extractfield(stamps,'Sec'));
    NanoSeconds = double(extractfield(stamps,'Nsec'));
    
    ori_x=extractfield(Orientation,'X')';
    ori_y=extractfield(Orientation,'Y')';
    ori_z=extractfield(Orientation,'Z')';
    ori_w=extractfield(Orientation,'W')';
    
    vel_ang_x=extractfield(Angular_velocity,'X')';
    vel_ang_y=extractfield(Angular_velocity,'Y')';
    vel_ang_z=extractfield(Angular_velocity,'Z')';
    
    acc_x=extractfield(Linear_acceleration,'X')';
    acc_y=extractfield(Linear_acceleration,'Y')';
    acc_z=extractfield(Linear_acceleration,'Z')';
    
    time = (Seconds + NanoSeconds*1e-9)';
    
    % CSV output
    csv=[time,ori_x,ori_y,ori_z,ori_w,vel_ang_x,vel_ang_y,vel_ang_z,acc_x,acc_y,acc_z];
    T = array2table(csv,'VariableNames',{'time','ori_x','ori_y','ori_z','ori_w','vel_ang_x','vel_ang_y','vel_ang_z','acc_x','acc_y','acc_z'});
    writetable(T,[csv_directory,'/',filename,'.csv']);
    
end

disp("DONE");