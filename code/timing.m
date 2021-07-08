% Parameter:
FileName = 'test.txt';
Key      = 'F-measure';
NewFile  = 'Fixed.txt';
% Import text file and select lines starting with the Key string:
Str   = fileread(FileName);
CStr  = strsplit(Str, '\n');
Match = strncmp(CStr, Key, length(Key));
CStr  = CStr(Match);
% Create new file and write matching lines:
fid = fopen(NewFile, 'w');
if fid == -1
  error('Cannot create new file: %s', NewFile);
end
fprintf(fid, '%s\n', CStr{:});
fclose(fid);

fid = fopen( 'Fixed.txt');
data = textscan(fid, '%s%s%s%s%s%s', 'Delimiter', '&','TreatAsEmpty','~');
fclose(fid);

f=data{2};
sorted=sort(f);
ordedR=flip(sorted);