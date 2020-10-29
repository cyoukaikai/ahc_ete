user_name = 'test'

% read xml file to structure
xml_path = 'annotation_setting.xml';

function extract_user_infor(xml_path, user_name)
user = struct([]) % creates an empty (0-by-0) structure with no fields.
user_infor = xml2struct(xml_path);

users = user_infor.annotation.user;
for k = 1 : length(users)
    if users{k}.Attributes.name == user_name
        user = users{k}
        break

return user




