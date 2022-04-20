CLASSES = [
    'Hip_DepuySynthes_Corail_Collar',
    'Hip_DepuySynthes_Corail_NilCol',
    'Hip_JRIOrtho_FurlongEvolution_Collar',
    'Hip_JRIOrtho_FurlongEvolution_NilCol',
    'Hip_SmithAndNephew_Anthology',
    'Hip_SmithAndNephew_Polarstem_NilCol',
    'Hip_Stryker_AccoladeII',
    'Hip_Stryker_Exeter',
    'Knee_Depuy_Synthes_Sigma',
    'Knee_SmithAndNephew_GenesisII',
    'Knee_SmithAndNephew_Legion2',
    'Knee_ZimmerBiomet_Oxford']


def format_prosthesis_name(prosthesis_code):
    implant_type, manufacturer = prosthesis_code.split('_')[:2]
    model_name = prosthesis_code.split('_', 2)[2].replace('_', ' ')
    prosthesis_name = f"{manufacturer} - {model_name} ({implant_type})"
    return prosthesis_name
