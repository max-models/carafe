// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/carafe/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'carafe',
	title: 'carafe',
	description: 'Global (0D) and 1D fluid models of low-temperature plasmas.',
	pythonPackage: 'carafe',
	// the method cards: a Numerics section with the overview (numerics/index.mdx) and one page per
	// module, in the order of the model, and a Numerics section on each method's API page
	methodcard: {
		pages: true,
		index: false,
		labels: {
			'carafe.reactions': 'Reactions',
			'carafe.global_model': 'Global model',
			'carafe.cr': 'OpenCR rates',
			'carafe.neutralizer': 'Neutralizer',
			'carafe.driver': 'RF driver (0D)',
			'carafe.magnetron': 'Magnetron',
			'carafe.sputtering': 'Sputter yields',
			'carafe.irm': 'Ionization region model',
			'carafe.fluid1d': 'Fluid 1D',
			'carafe.spatial.grid': 'Cell grid',
			'carafe.spatial.fluid': 'Fluid 2D',
			'carafe.spatial.inductive': 'Inductive field 2D',
			'carafe.spatial.edges': 'Edge grid',
			'carafe.spatial.cutcell': 'Cut cells',
			'carafe.spatial.inductive3d': 'Inductive field 3D',
			'carafe.spatial.iterative': 'Iterative solver',
			'carafe.spatial.coupling': 'Field and plasma',
			'carafe.spatial.neutrals': 'Neutral gas 2D',
			'carafe.driftfluid2d': 'Drift fluid 2D',
			'carafe.dispersion': 'Local dispersion',
			'carafe.dispersion.eigenmodes': 'Nonlocal eigenmodes',
			'carafe.gasdynamics': 'Gas dynamics',
			'carafe.gas_coupling': 'Gas and drift fluid',
			'carafe.transport': 'Transport budget',
		},
	},
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Tutorials', link: '/tutorials/' },
		{ label: 'Full examples', link: '/full-examples/' },
		{ label: 'Numerics', link: '/numerics/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/carafe/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
				{ label: 'Live demo', slug: 'getting-started/live-demo' },
			],
		},
		{
			label: 'Tutorials',
			items: [{ autogenerate: { directory: 'tutorials' } }],
		},
		{
			label: 'Full examples',
			items: [{ autogenerate: { directory: 'full-examples' } }],
		},
		{
			label: 'Development',
			items: [
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
				{ label: '3D field discretisation', slug: 'development/field-3d' },
			],
		},
	],
});
