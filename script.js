function change_tab(new_tab)
{
        document.getElementById('tab_'+current_tab).className = 'tab_off tab';
        document.getElementById('tab_'+new_tab).className = 'tab_on tab';
        document.getElementById('tab_content_'+current_tab).style.display = 'none';
        document.getElementById('tab_content_'+new_tab).style.display = 'block';
        current_tab = new_tab;
        return false;
}

var current_tab = 'publications';
var hash = window.location.hash.substr(1);
debugger;
if (hash == '')
    change_tab(current_tab);
else
    change_tab(hash);
