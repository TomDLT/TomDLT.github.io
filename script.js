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
        update_year_visibility();
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

// Hide / show the year rows so that only years containing at least one
// currently–visible publication remain on screen.
// First make all years visible, then hide those without any visible publications.
function update_year_visibility() {
    const $rows = $('#tab_content_publications > .row');
    $rows.show();
    $rows.each(function () {
        if ($(this).find('.pub:visible').length === 0) {
            $(this).hide();
        }
    });
}

// Call update_year_visibility every time the publication filter changes.
// We achieve this by wrapping the existing `select_publication` function.
if (typeof window.select_publication === 'function') {
    const originalSelect = window.select_publication;
    window.select_publication = function (pub_type = 'pub') {
        originalSelect(pub_type);      // run original logic
        update_year_visibility();      // refresh year visibility
    };
}

var current_tab = 'about_me';
var hash = window.location.hash.substr(1);
var url = window.location.href;

if (hash == '')
    change_tab(current_tab);
else {
    change_tab(hash);
}
