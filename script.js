function change_tab(new_tab)
{
        document.getElementById('tab_'+current_tab).className = 'tab_off tab';
        document.getElementById('tab_'+new_tab).className = 'tab_on tab';
        document.getElementById('tab2_'+current_tab).className = 'tab_off tab';
        document.getElementById('tab2_'+new_tab).className = 'tab_on tab';
        document.getElementById('tab_content_'+current_tab).style['display'] = 'none';
        document.getElementById('tab_content_'+new_tab).style['display'] = 'block';
        window.location.hash = new_tab;
        current_tab = new_tab;
        return false;
}

function select_publication(pub_type='pub') {
    var i;

    // Change button color
    var all_btn = document.getElementsByClassName("btn-secondary");
    for (i = 0; i < all_btn.length; i++) {
        all_btn[i].className = 'btn btn-secondary';
    }
    document.getElementById("btn_"+pub_type).className = 'btn btn-secondary btn_on';

    // Find selected publications
    var x = document.getElementById("tab_content_publications");
    var all = x.getElementsByClassName("pub");
    var except = x.getElementsByClassName(pub_type);

    // Update the visibility state of each publications
    for (i = 0; i < all.length; i++) {
        all[i].style['display'] = 'none';
    }
    for (i = 0; i < except.length; i++) {
        except[i].style['display'] = '';
    }
}

var current_tab = 'about_me';
var hash = window.location.hash.substr(1);

if (hash == '')
    change_tab(current_tab);
else
    change_tab(hash);

