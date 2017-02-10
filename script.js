function change_tab(new_tab)
{
        document.getElementById('tab_'+current_tab).className = 'tab_off tab';
        document.getElementById('tab_'+new_tab).className = 'tab_on tab';
        document.getElementById('tab_content_'+current_tab).style.display = 'none';
        document.getElementById('tab_content_'+new_tab).style.display = 'block';
        current_tab = new_tab;
        details(false);
}

var current_tab = 'publications';
change_tab(current_tab);
var show_details = false;
details(false);


function len(tab)
{
    switch(tab) {
    case 'experience':
        return 4
    case 'education':
        return 3
    case 'software':
        return 1
    default:
        return 0
}
}

function details(change)
{
    if (show_details == change)
    {
        show_details = false;
        document.getElementById('details_button').innerHTML = 'Show details';
        for (i = 0; i < len(current_tab); i++)
        {
            document.getElementById('details_'+current_tab+i).style.display = 'none'
        }
    }
    else
    {
        show_details = true;
        document.getElementById('details_button').innerHTML = 'Hide details';
        for (i = 0; i < len(current_tab); i++)
        {
            document.getElementById('details_'+current_tab+i).style.display = 'block'
        }
    }
}
